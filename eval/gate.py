"""baseline regression gate.

Compares the latest evaluation_report.json against a committed baseline (eval/baseline.json)
and exits non-zero if any metric regressed beyond --tolerance. This is the check CI
runs to block a merge that makes retrieval/answer quality worse.

  python -m eval.gate              # compare current report vs baseline; exit 1 on regression
  python -m eval.gate --update     # bless the current report as the new baseline
  python -m eval.gate --selftest   # verify the comparison logic
"""
import argparse
import json
import sys
from pathlib import Path

REPORT_PATH = Path("evaluation_report.json")
BASELINE_PATH = Path(__file__).parent / "baseline.json"

# metric -> higher_is_better
DIRECTION = {
    "faithfulness": True,
    "context_precision": True,
    "context_recall": True,
    "answer_relevancy": True,
    "abstention_accuracy": True,
    "hallucination_rate": False,
}


def find_regressions(baseline, current, tol):
    """Return [(metric, base, cur, delta), ...] for metrics that regressed beyond tol."""
    out = []
    for m, higher_better in DIRECTION.items():
        b, c = baseline.get(m), current.get(m)
        if not _num(b) or not _num(c):
            continue  # missing on one side -> can't compare
        delta = c - b
        if (delta < -tol) if higher_better else (delta > tol):
            out.append((m, b, c, delta))
    return out


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.05, help="allowed metric drop before failing")
    ap.add_argument("--update", action="store_true", help="write current metrics as the new baseline")
    ap.add_argument("--selftest", action="store_true", help="check the comparison logic and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        print("selftest OK")
        return

    if not REPORT_PATH.exists():
        sys.exit(f"no {REPORT_PATH}; run `python -m eval.runner` first")
    current = _load(REPORT_PATH)["metrics"]

    if args.update:
        BASELINE_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"baseline updated -> {BASELINE_PATH}")
        return

    if not BASELINE_PATH.exists():
        sys.exit(f"no baseline at {BASELINE_PATH}; create one with: python -m eval.gate --update")
    baseline = _load(BASELINE_PATH)
    regressions = find_regressions(baseline, current, args.tolerance)

    print(f"{'metric':22}{'baseline':>10}{'current':>10}{'delta':>10}")
    for m in DIRECTION:
        b, c = baseline.get(m), current.get(m)
        bs = f"{b:.4f}" if _num(b) else "—"
        cs = f"{c:.4f}" if _num(c) else "—"
        ds = f"{c - b:+.4f}" if _num(b) and _num(c) else "—"
        flag = "  <-- REGRESSED" if any(r[0] == m for r in regressions) else ""
        print(f"{m:22}{bs:>10}{cs:>10}{ds:>10}{flag}")

    if regressions:
        sys.exit(f"\nGATE FAILED: {len(regressions)} metric(s) regressed > {args.tolerance}")
    print(f"\nGATE PASSED (tolerance {args.tolerance})")


def _selftest():
    base = {"faithfulness": 0.9, "hallucination_rate": 0.1}
    assert find_regressions(base, {"faithfulness": 0.8, "hallucination_rate": 0.1}, 0.05)   # quality drop
    assert find_regressions(base, {"faithfulness": 0.9, "hallucination_rate": 0.2}, 0.05)   # more hallucination
    assert not find_regressions(base, {"faithfulness": 0.87, "hallucination_rate": 0.12}, 0.05)  # within tol
    assert not find_regressions(base, {"faithfulness": 0.95, "hallucination_rate": 0.05}, 0.05)  # improved


if __name__ == "__main__":
    main()
