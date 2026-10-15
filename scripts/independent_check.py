"""Independent cross-check of output.csv.

Recomputes weekly cost, own-history and peer deviations, and the flag set from
shipment_records.csv using a separate, minimal implementation (no src/ imports),
then compares against output.csv. Run from the repo root:
    python scripts/independent_check.py
"""
import re
import sys
import numpy as np
import pandas as pd

T_OWN, T_PEER, WINDOW = 6.0, 20.0, 8

d = pd.read_csv("shipment_records.csv")
d["route"] = d.origin + "-" + d.destination
dt = pd.to_datetime(d.shipment_date)
d["week_of"] = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()
d["tonne_km"] = d.quantity_tonnes * d.distance_km
g = (d.groupby(["route", "route_type", "week_of"])
       .agg(cost=("freight_cost_inr", "sum"), tk=("tonne_km", "sum"))
       .reset_index())
g["unit"] = g.cost / g.tk

rows = []
for route, grp in g.groupby("route"):
    grp = grp.sort_values("week_of").reset_index(drop=True)
    for i in range(1, len(grp)):
        base = grp.unit[max(0, i - WINDOW):i].mean()   # only strictly-prior weeks
        peers = g[(g.route_type == grp.route_type[i]) & (g.week_of == grp.week_of[i])
                  & (g.route != route)].unit
        peer = (grp.unit[i] / peers.mean() - 1) * 100 if len(peers) else np.nan
        rows.append((route, grp.week_of[i], grp.unit[i], (grp.unit[i] / base - 1) * 100, peer))
x = pd.DataFrame(rows, columns=["route", "week_of", "unit", "own", "peer"])
exp = x[(x.own > 0) & ((x.own >= T_OWN) | (x.peer >= T_PEER))]

out = pd.read_csv("output.csv")
out["week_of"] = pd.to_datetime(out.week_of)
m = exp.merge(out, on=["route", "week_of"], how="outer", indicator=True)
only = m[m._merge != "both"]
both = m[m._merge == "both"].copy()
own_out = both.vs_own_history.str.extract(r"([+-][\d.]+)%")[0].astype(float)
peer_out = both.vs_similar_routes.str.extract(r"([+-][\d.]+)%")[0].astype(float)

checks = {
    "route-weeks evaluated": (len(x), 721),
    "flag set identical (rows only on one side)": (len(only), 0),
    "max |cost diff| (rounded 2dp)": (float((both.unit.round(2) - both.cost_per_tonne_km).abs().max()), 0.0),
    "max |own % diff| (rounded 1dp)": (float((both.own.round(1) - own_out).abs().max()), 0.0),
    "max |peer % diff| (rounded 1dp)": (float((both.peer.round(1) - peer_out).abs().max()), 0.0),
}
ok = True
for name, (got, want) in checks.items():
    good = got == want
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: {got}")
sys.exit(0 if ok else 1)
