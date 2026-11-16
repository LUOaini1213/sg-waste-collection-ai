-- KPI tables (DuckDB). Raw tables are registered by wastecollect/pipeline/kpis.py:
--   daily_ops(scenario, seed, policy, day, dow, bins, load, routes, unloads, km, feasible,
--             overflow_bins, overflow_volume, mean_fill_at_collection)
--   sim_meta(warmup_days, co2_kg_per_km)
--   fill_levels(bin, date, eod, max, mean, changes, coll, stale), fill_bins(bin, waste_type, ...)
--   forecast_metrics(method, horizon, mae, rmse, bias, n)
--   routing(instance, customers, best, mean, reference, reference_source, gap_best_pct, gap_mean_pct)

-- 1. policy totals over the measured period (mean over seeds), with change versus the fixed schedule
CREATE OR REPLACE TABLE kpi_policy AS
WITH per_seed AS (
    SELECT d.scenario, d.seed, d.policy,
           SUM(d.km)                              AS km,
           SUM(d.bins)                            AS lifts,
           SUM(d.routes)                          AS truck_shifts,
           MAX(d.routes)                          AS max_trucks_per_day,
           SUM(d.overflow_bins)                   AS overflow_bin_days,
           SUM(d.overflow_volume)                 AS overflow_volume,
           SUM(d.load)                            AS loads_collected,
           SUM(d.km) * MAX(m.co2_kg_per_km)       AS co2_kg,
           BOOL_AND(d.feasible)                   AS all_feasible
    FROM daily_ops d CROSS JOIN sim_meta m
    WHERE d.day >= m.warmup_days
    GROUP BY d.scenario, d.seed, d.policy
), agg AS (
    SELECT scenario, policy, COUNT(*) AS seeds,
           AVG(km) AS km, AVG(lifts) AS lifts, AVG(truck_shifts) AS truck_shifts,
           MAX(max_trucks_per_day) AS max_trucks_per_day, AVG(overflow_bin_days) AS overflow_bin_days,
           AVG(overflow_volume) AS overflow_volume, AVG(loads_collected) AS loads_collected,
           AVG(co2_kg) AS co2_kg, BOOL_AND(all_feasible) AS all_feasible
    FROM per_seed GROUP BY scenario, policy
)
SELECT a.*,
       a.km / NULLIF(a.lifts, 0)                       AS km_per_lift,
       a.loads_collected / NULLIF(a.lifts, 0)          AS mean_fill_at_lift,
       100.0 * (a.km - f.km) / NULLIF(f.km, 0)         AS km_change_pct,
       100.0 * (a.lifts - f.lifts) / NULLIF(f.lifts, 0) AS lifts_change_pct,
       100.0 * (a.overflow_bin_days - f.overflow_bin_days) / NULLIF(f.overflow_bin_days, 0) AS overflow_change_pct,
       100.0 * (a.co2_kg - f.co2_kg) / NULLIF(f.co2_kg, 0) AS co2_change_pct
FROM agg a JOIN agg f ON f.scenario = a.scenario AND f.policy = 'fixed'
ORDER BY a.scenario, a.policy;

-- 2. weekly series for the dashboard (measured period, mean over seeds)
CREATE OR REPLACE TABLE kpi_weekly AS
SELECT d.scenario, d.policy, (d.day - m.warmup_days) // 7 + 1 AS week,
       SUM(d.km) / COUNT(DISTINCT d.seed)            AS km,
       SUM(d.bins) / COUNT(DISTINCT d.seed)          AS lifts,
       SUM(d.overflow_bins) / COUNT(DISTINCT d.seed) AS overflow_bin_days
FROM daily_ops d CROSS JOIN sim_meta m
WHERE d.day >= m.warmup_days
GROUP BY d.scenario, d.policy, week
ORDER BY d.scenario, d.policy, week;

-- 3. observed smart-bin behaviour in Trnava by waste type (stale sensor days excluded)
CREATE OR REPLACE TABLE kpi_bins_observed AS
WITH x AS (
    SELECT l.*, b.waste_type,
           l.eod - LAG(l.eod) OVER (PARTITION BY l.bin ORDER BY l.date) AS inc,
           LAG(l.stale) OVER (PARTITION BY l.bin ORDER BY l.date)       AS prev_stale
    FROM fill_levels l JOIN fill_bins b USING (bin)
)
SELECT waste_type,
       COUNT(DISTINCT bin)                                                    AS bins,
       COUNT(*)                                                               AS bin_days,
       AVG(CASE WHEN stale = 0 THEN eod END)                                  AS mean_level_pct,
       AVG(CASE WHEN stale = 0 AND prev_stale = 0 AND coll = 0 AND inc > -20 THEN inc END) AS mean_daily_increment_pp,
       7.0 * SUM(coll) / NULLIF(SUM(CASE WHEN stale = 0 THEN 1 END), 0)       AS collections_per_bin_week,
       AVG(CASE WHEN stale = 0 THEN (max >= 90)::INT END)                     AS share_days_max_ge_90,
       AVG(stale::DOUBLE)                                                     AS stale_share
FROM x
GROUP BY waste_type
ORDER BY waste_type;

-- 4. forecast error table, best method per horizon
CREATE OR REPLACE TABLE kpi_forecast AS
SELECT f.*, f.mae = MIN(f.mae) OVER (PARTITION BY f.horizon) AS best_in_horizon
FROM forecast_metrics f
ORDER BY f.horizon, f.mae;

-- 5. routing benchmark gaps
CREATE OR REPLACE TABLE kpi_routing AS
SELECT *, CASE WHEN gap_best_pct <= 0 THEN 'at or below reference' WHEN gap_best_pct <= 2 THEN 'within 2%'
               WHEN gap_best_pct <= 5 THEN 'within 5%' ELSE 'above 5%' END AS band
FROM routing
ORDER BY customers;
