#!/usr/bin/env python3
"""Self-test for main.py: format, determinism, and practice accuracy.

Usage:
    python3 tests/selftest.py <kit-input-dir>

Runs main.py twice on the kit's practice cases and checks that
  1. every case gets a well-formed <id>.out (exactly one forecast row,
     price a finite number in (0, 10_000_000));
  2. the two runs write byte-identical outputs (the grader runs each case
     twice and requires identical bytes);
  3. on the five practice cases whose answers are recoverable (each case
     p0k's target is the next case's last close), the mean absolute error
     beats the persistence baseline's.

This test is practice-only and ships no data: the kit's input/ directory
is passed in, never committed.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent / "main.py"


def run_once(input_dir, output_dir):
    env = {"MAKEFUN_INPUT_DIR": str(input_dir), "MAKEFUN_OUTPUT_DIR": str(output_dir)}
    subprocess.run(
        [sys.executable, str(MAIN)], env=env, check=True, capture_output=True,
        timeout=120,
    )


def main():
    input_dir = Path(sys.argv[1])
    cases = {}
    for path in sorted(input_dir.glob("*.in")):
        with path.open() as fh:
            cases[path.stem] = json.load(fh)["markets"][0]["prices"]
    assert cases, f"no .in files in {input_dir}"

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        run_once(input_dir, d1)
        run_once(input_dir, d2)
        for cid in cases:
            o1 = Path(d1, f"{cid}.out").read_bytes()
            o2 = Path(d2, f"{cid}.out").read_bytes()
            assert o1 == o2, f"{cid}: two runs differ"
            out = json.loads(o1)
            assert set(out) == {"forecasts"}, f"{cid}: bad top-level keys"
            rows = out["forecasts"]
            assert len(rows) == 1, f"{cid}: expected one forecast row"
            row = rows[0]
            assert set(row) == {"case", "price"}, f"{cid}: bad row keys"
            assert row["case"] == cid, f"{cid}: row names another case"
            p = row["price"]
            assert isinstance(p, (int, float)) and math.isfinite(p), f"{cid}: bad price"
            assert 0 < p < 10_000_000, f"{cid}: price out of bounds"
        print(f"format + determinism OK ({len(cases)} cases)")

        # Practice accuracy: p0k's target is p0(k+1)'s last close.
        ids = sorted(cases)
        errs, base = [], []
        for a, b in zip(ids, ids[1:]):
            pred = json.loads(Path(d1, f"{a}.out").read_bytes())["forecasts"][0]["price"]
            target = cases[b][-1]
            errs.append(100 * abs(pred - target))
            base.append(100 * abs(cases[a][-1] - target))
        if errs:
            mae = sum(errs) / len(errs)
            bmae = sum(base) / len(base)
            print(f"practice MAE: {mae:.0f} cents vs persistence {bmae:.0f} cents "
                  f"({len(errs)} cases)")
            assert mae < bmae, "does not beat persistence on practice"
    print("selftest PASSED")


if __name__ == "__main__":
    main()
