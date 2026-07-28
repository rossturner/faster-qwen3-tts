"""Experiment 3, Stage A: corrected analysis of the inertness runs.

The pre-registered rule in the spec was mis-specified. It compared the spread of
condition *means* against the spread of *individual runs* -- but a mean of 5 runs varies
by std/sqrt(5), so the rule demanded an effect ~2.2x larger than it should have. It
called the positive control inert, which is how the error was caught: the control's raw
numbers (whispered RMS 0.073 vs 0.11, excited F0 170Hz vs 133Hz) are unmistakable.

This applies the standard test for the question actually being asked -- one-way ANOVA
across conditions, F = MS_between / MS_within, with the same data. The *statistic* is
corrected; the threshold (p < 0.05) is conventional and fixed here before looking at
the clone results. The control validates it: if the control does not come out clearly
significant, this analysis is still wrong and no conclusion may be drawn.

Reads inertness_results.json -- no regeneration.

Usage:
    .venv/bin/python spikes/emotion/reanalyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METRICS = ["duration_s", "chars_per_s", "rms", "f0_median", "voiced_frac"]
# F critical, alpha=0.05, df=(3,16): 4 conditions x 5 runs
F_CRIT = 3.24


def anova(groups: list[list[float]]) -> tuple[float, float]:
    """One-way ANOVA F statistic and the between/within df."""
    groups = [np.asarray([v for v in g if not np.isnan(v)], dtype=float) for g in groups]
    groups = [g for g in groups if len(g) > 1]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = k - 1, n_total - k
    if ss_within == 0 or df_w == 0:
        return float("inf"), df_w
    return (ss_between / df_b) / (ss_within / df_w), df_w


def analyse(label: str, data: dict) -> dict:
    print(f"\n  {label}")
    print(f"    {'metric':<14} {'F':>8}  {'sig?':>5}   condition means")
    verdicts = {}
    for metric in METRICS:
        groups = [[r[metric] for r in runs] for runs in data.values()]
        f, _ = anova(groups)
        sig = f > F_CRIT
        verdicts[metric] = {"F": round(f, 2), "significant": bool(sig)}
        means = "  ".join(f"{name}={np.nanmean([r[metric] for r in runs]):.3g}"
                          for name, runs in data.items())
        print(f"    {metric:<14} {f:>8.2f}  {'YES' if sig else '-':>5}   {means}")
    return verdicts


def main() -> None:
    path = Path(__file__).parent / "inertness_results.json"
    payload = json.loads(path.read_text())

    print(f"One-way ANOVA across 4 instruct conditions, 5 runs each "
          f"(F critical = {F_CRIT}, alpha=0.05)")
    control_v = analyse("CONTROL (CustomVoice/aiden — instruct officially supported)",
                        payload["control"])
    clone_v = analyse("CLONE (en_f, ICL — the path we would ship)", payload["clone"])

    control_ok = any(v["significant"] for v in control_v.values())
    clone_metrics = [m for m, v in clone_v.items() if v["significant"]]

    print("\n  VERDICT")
    if not control_ok:
        print("    INVALID — the control shows no effect, so this analysis cannot")
        print("    distinguish an inert engine from an insensitive measurement.")
    elif clone_metrics:
        print(f"    instruct MOVES the clone path, on: {', '.join(clone_metrics)}")
        print(f"    but NOT on: {', '.join(m for m in METRICS if m not in clone_metrics)}")
        print("    -> partial instruction-following. Whether it is enough for emotion is")
        print("       a listening question, not a statistical one.")
    else:
        print("    instruct is INERT on the clone path, while the control responds.")
        print("    -> Stage B must use reference clips.")

    (Path(__file__).parent / "reanalysis.json").write_text(
        json.dumps({"control": control_v, "clone": clone_v}, indent=2))


if __name__ == "__main__":
    main()
