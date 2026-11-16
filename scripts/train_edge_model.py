"""Fine-tune MobileNetV3-Small on TrashNet and export it to ONNX (separate, documented step).

Needs torch + torchvision + onnx (requirements-edge.txt) and the images from
scripts/fetch_trashnet.py. Uses CUDA when available. Writes
  data/inputs/trashnet_split.csv        the fixed 70/13/17 stratified split (file names only)
  models/trashnet_mnv3s_fp32.onnx       FP32 model, opset 17, dynamic batch
  results/edge_training.json            epochs, validation curve, device, versions, wall time
Training is seeded but GPU kernels are not bit-reproducible; the committed ONNX file and the
metrics computed from it (scripts/quantize_edge_model.py) are the reference.

usage: python scripts/train_edge_model.py [--epochs 25]
"""
from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wastecollect.edge.data import CLASSES, list_images, make_split  # noqa: E402
from wastecollect.io import write_csv, write_json  # noqa: E402

IMG = ROOT / "data" / "external" / "trashnet" / "dataset-resized"


def main(argv=None) -> int:
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torchvision import transforms as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = make_split(list_images(IMG), seed=args.seed)
    write_csv(ROOT / "data" / "inputs" / "trashnet_split.csv", df[["file", "class", "label", "split"]])
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    tf_eval = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])

    class DS(torch.utils.data.Dataset):
        def __init__(self, frame, tf):
            self.files = frame["file"].tolist()
            self.labels = frame["label"].tolist()
            self.tf = tf
            self.cache = {}

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            im = self.cache.get(i)
            if im is None:
                with Image.open(IMG / self.files[i]) as f:
                    im = f.convert("RGB")
                self.cache[i] = im
            return self.tf(im), self.labels[i]

    g = torch.Generator().manual_seed(args.seed)
    dl_train = torch.utils.data.DataLoader(DS(df[df.split == "train"], tf_train), batch_size=args.batch,
                                           shuffle=True, num_workers=0, generator=g)
    dl_val = torch.utils.data.DataLoader(DS(df[df.split == "val"], tf_eval), batch_size=64, num_workers=0)
    dl_test = torch.utils.data.DataLoader(DS(df[df.split == "test"], tf_eval), batch_size=64, num_workers=0)

    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def evaluate(dl):
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in dl:
                p = model(x.to(dev)).argmax(1).cpu()
                correct += int((p == y).sum())
                n += len(y)
        return correct / n

    t0 = time.perf_counter()
    curve, best, best_state = [], -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = cnt = 0.0
        for x, y in dl_train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            cnt += len(y)
        sched.step()
        va = evaluate(dl_val)
        curve.append({"epoch": ep, "train_loss": tot / cnt, "val_acc": va})
        print(f"epoch {ep:2d} loss {tot / cnt:.4f} val_acc {va:.4f}", flush=True)
        if va > best:
            best, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    test_acc_torch = evaluate(dl_test)
    model = model.cpu().eval()
    out = ROOT / "models" / "trashnet_mnv3s_fp32.onnx"
    out.parent.mkdir(exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), str(out), opset_version=17, input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    write_json(ROOT / "results" / "edge_training.json", {
        "model": "MobileNetV3-Small (torchvision, ImageNet-1k weights IMAGENET1K_V1), new 6-class head",
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "optimizer": "AdamW, weight decay 1e-4, cosine schedule, label smoothing 0.1",
        "augmentation": "RandomResizedCrop(224, scale 0.5-1), horizontal and vertical flips, ColorJitter(0.2)",
        "best_epoch": best_epoch, "best_val_acc": best, "test_acc_pytorch": test_acc_torch,
        "train_seconds": train_s, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor(),
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "python": platform.python_version()},
        "split_counts": df["split"].value_counts().to_dict(), "curve": curve})
    print("best epoch", best_epoch, "val", best, "test (pytorch)", test_acc_torch, f"{train_s:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
