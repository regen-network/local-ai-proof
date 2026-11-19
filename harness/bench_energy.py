"""Rough energy per record: whole-machine power above idle while a command repeats.

  python3 harness/bench_energy.py --records 22 --repeat 30 --label hybrid-reg -- python3 harness/rank.py ...

Reads macOS battery telemetry (no root). Other workloads on the machine add noise: the idle
baseline is taken just before, and the report keeps the raw numbers so the noise is visible.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

from common import PowerSampler, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, required=True, help="records processed per command run")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--label", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    ps = PowerSampler()
    idle = ps.baseline(10)
    with ps:
        for _ in range(a.repeat):
            subprocess.run(cmd, check=True, capture_output=True)
    e = ps.joules(idle)
    n = a.records * a.repeat
    after_idle = PowerSampler().baseline(5)
    out = {"label": a.label, "records": n, **e, "idle_after_w": round(after_idle / 1000, 2) if after_idle else None,
           "seconds_per_record": round(e["seconds"] / n, 4),
           "joules_above_idle_per_record": round(e.get("above_idle_j", 0) / n, 3),
           "wh_above_idle_per_1000_records": round(e.get("above_idle_j", 0) / n * 1000 / 3600, 3)}
    write_json(f"private/runs/energy-{a.label}.json", out)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
