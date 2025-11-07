from fastapi import FastAPI, UploadFile, File, Form


# Pagrindiniai agregatai
total_produced = df["produced_qty"].sum()
total_good = df["good_qty"].sum()
total_scrap = df["scrap_qty"].sum()


# Downtime ir ciklo laikas
total_downtime_min = df["downtime_min"].fillna(0).sum()
avg_cycle_time = df["cycle_time_sec"].replace(0, pd.NA).dropna().mean()
cycle_time_sec = float(avg_cycle_time) if pd.notna(avg_cycle_time) else 0.0


# Performance/Throughput
hours = max((len(df) * cycle_time_sec) / 3600.0, 0.0001)
throughput_per_hour = float(total_produced / hours)


# Quality
quality = float(total_good / max(total_produced, 1))


# Availability: laikom, kad planuotas laikas = faktinis laikas + downtime
planned_time_hours = hours + (total_downtime_min / 60.0)
availability = float(hours / max(planned_time_hours, 0.0001))


# Performance: idealus CT * output / operating time
ideal_ct = df["cycle_time_sec"].replace(0, pd.NA).dropna().min()
if pd.isna(ideal_ct):
ideal_ct = cycle_time_sec if cycle_time_sec > 0 else 1.0
performance = float((total_produced * ideal_ct) / max(hours * 3600.0, 0.0001))
performance = min(max(performance, 0.0), 1.5) # apribojam outlier'ius


# OEE
oee = availability * performance * quality


# Scrap rate
scrap_rate = float(total_scrap / max(total_produced, 1))


# Top downtime priežastys (jei yra)
top_downtime = []
if "downtime_reason" in df.columns:
top = (
df.groupby("downtime_reason")["downtime_min"].sum()
.sort_values(ascending=False)
.head(5)
)
top_downtime = [
{"reason": r, "minutes": float(m)} for r, m in top.items()
]


# Butelio kakliukas – mažiausio našumo stotis
bottleneck_station = None
if "station" in df.columns and "cycle_time_sec" in df.columns:
perf_by_station = (
df.groupby("station")["cycle_time_sec"].mean().sort_values(ascending=False)
)
if len(perf_by_station) > 0:
bottleneck_station = str(perf_by_station.index[0])


# Potencialios mėnesio sutaupytos sąnaudos, jei pasiekiamas target OEE
current_oee = oee
improvement = max(target_oee - current_oee, 0.0)


# apytiksliai: per dieną pagaminto gero kiekio vertė * OEE pagerėjimas
good_per_day = (total_good / max(len(df), 1)) * (shift_hours * 3600.0 / max(cycle_time_sec or 1.0, 1.0))
potential_savings = good_per_day * unit_cost_eur * improvement * workdays_per_month


return AnalyzeResponse(
oee=round(oee, 4),
availability=round(availability, 4),
performance=round(performance, 4),
quality=round(quality, 4),
cycle_time_sec=round(cycle_time_sec, 2),
throughput_per_hour=round(throughput_per_hour, 2),
scrap_rate=round(scrap_rate, 4),
top_downtime=top_downtime,
bottleneck_station=bottleneck_station,
potential_savings_eur_per_month=round(float(potential_savings), 2),
)
