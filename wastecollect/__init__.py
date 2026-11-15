"""AI and optimisation for urban waste collection (personal desk study).

Sub-packages:
  routing    - waste-collection VRP with intermediate facilities (IF), time windows and a driver break
  forecast   - bin fill-level forecasting backtests
  casestudy  - Singapore HDB-town "collect when needed" vs fixed-schedule simulation
  edge       - waste-image classifier, ONNX export and INT8 quantisation (training needs torch)
  pipeline   - DuckDB KPI tables and a static Plotly dashboard
"""
__version__ = "0.1.0"
