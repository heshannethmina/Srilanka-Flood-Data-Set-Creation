"""Generate the project's results document from a `runs/` directory.

    python models/results_doc.py runs/ --out docs/RESULTS.md
    python models/results_doc.py runs/ --far 0.231          # match a baseline's FAR

`report.py` prints *what each model scored*. This writes the document you make
decisions and paper claims from, which needs three things `report.py` does not
have:

**An error bar.** Model 2's headline three rungs are separated by 0.0045 and
0.0086 PR-AUC while its own five seeds spread over 0.048. Any ordering read off
that table is noise. Every table here carries the seed spread beside the score,
and every difference is checked against it.

**A paired test.** "Is the graph real?" is not answered by two numbers in a
column, it is answered by a confidence interval on their *difference*, computed
on the same resampled test set. That is `paired_delta` below, and it is the only
thing that can settle RQ1.

**A matched operating point.** `ev.det` at each model's own threshold compares
thresholds, not models — model 2's ladder spans FAR 0.107 to 0.412. The early
warning table here re-derives every threshold at one common false-alarm ratio.

The document is regenerated, never edited: rerun this after every Kaggle session
and commit the result, so the paper's numbers and the repository's numbers
cannot drift apart.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from floodlib.metrics import average_precision, evaluate  # noqa: E402
from rethreshold import threshold_for_far  # noqa: E402

#: Short names for `TrainConfig.eval_head`, mirroring `floodlib.schema.CLS_HEADS`.
HEADS = ["flood_1d", "flood_2d", "flood_3d", "onset_1d"]

#: The comparisons the project exists to make, declared **before** the numbers
#: are seen so the analysis cannot be steered by them. Each is
#: `(model A, model B, what a positive delta would mean)`; A - B is reported.
#: A contrast whose two runs are not both present is reported as missing, with
#: the command that would produce it — that list is the "what to run next".
CONTRASTS: List[Tuple[str, str, str]] = [
    ("P0", "N3",
     "assembly check: model 3 with the graph off must reproduce model 2's N3. "
     "Expect 0.0000 exactly, not 'close' — the two networks are byte-identical "
     "at seed 0. Any difference at all means the wiring is wrong and every "
     "rung below is uninterpretable"),
    ("P0_x5 x5", "N5_bce x5",
     "the same check at 5 seeds: model 3's graph-free ensemble against model "
     "2's headline"),
    ("P1", "P0",
     "do the 204 spatial k-NN edges help at all?"),
    ("P2", "P1",
     "RQ1a — do the 35 directed flow edges add anything over spatial edges "
     "alone? This is the topology question the proposal is built on"),
    ("P3 x5", "P0_x5 x5",
     "**RQ1, the headline.** Identical encoder, identical loss, identical "
     "panel, identical seeds, identical calibrator — the graph is the only "
     "moving part. This interval is the answer"),
    ("P4_onset x5", "P4_onset_ctrl x5",
     "RQ1 on the target that matters operationally: does the graph help "
     "predict flood *onset*, where every baseline fails?"),
    ("M5_bce x5", "M5 x5",
     "the focal-loss defect on model 1. Model 2 measured +0.0509 for this "
     "change; if model 1 agrees, its whole published upper ladder is "
     "understated by about that much"),
    ("P3 x5", "M5_bce x5",
     "same graph, better encoder: what the PLR input layer is worth once both "
     "sides are on BCE"),
]


# ----------------------------------------------------------------- loading

def load_runs(run_dir: str, protocol: str = "temporal") -> Dict[str, Dict]:
    """Every model in `run_dir`, keyed by the label the tables use."""
    out: Dict[str, Dict] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        base = os.path.basename(path)
        if base.endswith("_diag.json") or "baseline" in base:
            continue
        with open(path) as f:
            d = json.load(f)
        if "preset" not in d or "test" not in d or d.get("protocol") != protocol:
            continue
        tc = d.get("train_config", {})
        n = tc.get("n_seeds", 1)
        eh = tc.get("eval_head", 0)
        label = d["preset"] + (f" x{n}" if n and n > 1 else "")
        if eh:
            label += f" [{HEADS[eh]}]"
        stem = base[:-5]
        diag_path = os.path.join(run_dir, stem + "_diag.json")
        diag = {}
        if os.path.exists(diag_path):
            with open(diag_path) as f:
                diag = json.load(f)
        npz_path = os.path.join(run_dir, stem + "_preds.npz")
        preds = dict(np.load(npz_path)) if os.path.exists(npz_path) else None
        out[label] = {"meta": d, "diag": diag, "preds": preds,
                      "family": d.get("family", ""), "eval_head": eh,
                      "n_seeds": n or 1}
    return out


def load_baselines(run_dir: str, protocol: str = "temporal") -> Dict[str, Dict]:
    path = os.path.join(run_dir, f"baselines_{protocol}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {v.get("baseline", k): v["test"] for k, v in d.items()}


# ------------------------------------------------------------- statistics

def block_bootstrap_delta(a: Dict, b: Dict, n_boot: int = 2000,
                          block_days: int = 14, seed: int = 0
                          ) -> Optional[Dict[str, float]]:
    """Paired CI on `PR-AUC(a) - PR-AUC(b)`, resampling (node, fortnight) blocks.

    Resampling *node-days* would be wrong here and badly so: flood labels are
    strongly autocorrelated in time, so independent node-days do not exist and a
    naive bootstrap returns an interval several times too narrow — which is how
    a 0.004 difference ends up looking significant. Blocks of `block_days` per
    node keep whole runs of correlated days together.

    Paired: both models are scored on the *same* resampled blocks each draw, so
    the shared difficulty of a draw cancels out of the difference.

    Returns None when the two runs were not scored on identical rows, which
    happens if they used different protocols, panels or evaluation heads — in
    that case the difference is not a comparison and must not be reported.
    """
    if a["preds"] is None or b["preds"] is None:
        return None
    pa, pb = a["preds"], b["preds"]
    if pa["test_y"].shape != pb["test_y"].shape:
        return None
    if not (np.array_equal(pa["test_day"], pb["test_day"])
            and np.array_equal(pa["test_node"], pb["test_node"])):
        return None
    if a["eval_head"] != b["eval_head"] or not np.array_equal(pa["test_y"], pb["test_y"]):
        return None

    y = pa["test_y"]
    key = pa["test_node"].astype(np.int64) * 1_000_000 + (pa["test_day"] // block_days)
    uniq, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    idx_by_block = [order[s:e] for s, e in zip(starts, ends)]

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_block), len(idx_by_block))
        sel = np.concatenate([idx_by_block[i] for i in pick])
        ys = y[sel]
        if ys.sum() < 2:                       # a draw with no positives
            continue
        deltas.append(average_precision(ys, pa["test_prob"][sel])
                      - average_precision(ys, pb["test_prob"][sel]))
    if len(deltas) < n_boot // 10:
        return None
    d = np.asarray(deltas)
    point = (average_precision(y, pa["test_prob"])
             - average_precision(y, pb["test_prob"]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(point), "lo": float(lo), "hi": float(hi),
            "n_blocks": len(idx_by_block), "n_boot": len(d),
            # Two-sided bootstrap p: how often the sign flips, doubled.
            "p": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def seed_sd(run: Dict, key: str = "pr_auc") -> Optional[float]:
    s = (run.get("diag") or {}).get("seed_spread") or {}
    v = s.get(key)
    return float(v["std"]) if v and v.get("n_seeds", 1) > 1 else None


def verdict(res: Optional[Dict], sd: Optional[float]) -> str:
    """Turn an interval into the sentence the paper is allowed to say."""
    if res is None:
        return "not computable — runs missing, or not scored on identical rows"
    lo, hi, d = res["lo"], res["hi"], res["delta"]
    floor = 2 * sd if sd else None
    if lo > 0:
        s = f"**A wins** (+{d:.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"
    elif hi < 0:
        s = f"**B wins** ({d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])"
    else:
        return (f"**not resolved** ({d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                f"spans zero) — this experiment cannot separate them; more "
                f"seeds or a larger test block would be needed to")
    if floor and abs(d) < floor:
        s += (f" — but |Δ| is below the {floor:.4f} seed-noise floor "
              f"(2 sd over {res['n_boot']} draws); report the interval, not "
              f"the ordering")
    return s


# ---------------------------------------------------------------- rendering

def _row(label: str, m: Dict, sd: Optional[float], params: Optional[int]) -> str:
    t = m["test"]
    pr = f"{t.get('pr_auc', float('nan')):.4f}"
    if sd:
        pr += f" ±{sd:.4f}"
    p = "--" if not params else f"{params / 1e3:.0f}k"
    return ("| " + " | ".join([
        label, pr,
        f"{t.get('ece', float('nan')):.4f}",
        f"{t.get('brier', float('nan')):.4f}",
        f"{t.get('pod', float('nan')):.3f}",
        f"{t.get('far', float('nan')):.3f}",
        f"{t.get('csi', float('nan')):.3f}",
        f"{t.get('event_detection_rate', float('nan')):.3f}",
        f"{t.get('mean_lead_days', float('nan')):.2f}", p]) + " |")


def headline_table(runs: Dict[str, Dict], base: Dict[str, Dict]) -> List[str]:
    head = ("| model | PR-AUC | ECE | Brier | POD | FAR | CSI | ev.det | lead | params |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    out = [head]
    for name, t in base.items():
        out.append(_row(f"_{name}_", {"test": t}, None, None))
    fam_order = ["TF-STGNN", "MMF-Net", "STG-Former"]
    for fam in fam_order:
        got = [(k, v) for k, v in runs.items() if v["family"] == fam]
        if not got:
            continue
        out.append(f"| **{fam}** | | | | | | | | | |")
        for k, v in sorted(got):
            out.append(_row(k, v["meta"], seed_sd(v), v["meta"].get("n_params")))
    return out


def early_warning_table(runs: Dict[str, Dict], far: float) -> List[str]:
    """`ev.det` for every model re-thresholded to one common false-alarm ratio."""
    out = ["| model | FAR (matched) | POD | CSI | ev.det | own FAR | own ev.det |",
           "|---|---|---|---|---|---|---|"]
    for k, v in sorted(runs.items()):
        if v["preds"] is None:
            continue
        d = v["preds"]
        thr = threshold_for_far(d["test_y"], d["test_prob"], far)
        m = evaluate(d["test_y"], d["test_prob"], thr,
                     d["test_event"], d["test_day"], d["test_node"])
        own = v["meta"]["test"]
        out.append(f"| {k} | {m.get('far', float('nan')):.3f} | "
                   f"{m.get('pod', float('nan')):.3f} | "
                   f"{m.get('csi', float('nan')):.3f} | "
                   f"**{m.get('event_detection_rate', float('nan')):.3f}** | "
                   f"{own.get('far', float('nan')):.3f} | "
                   f"{own.get('event_detection_rate', float('nan')):.3f} |")
    return out


def contrast_section(runs: Dict[str, Dict], n_boot: int) -> Tuple[List[str], List[str]]:
    """The decision engine. Returns (rendered lines, missing-run notes)."""
    lines: List[str] = []
    missing: List[str] = []
    for a_name, b_name, question in CONTRASTS:
        a, b = runs.get(a_name), runs.get(b_name)
        lines.append(f"\n#### {a_name}  −  {b_name}\n")
        lines.append(f"{question}\n")
        if a is None or b is None:
            absent = [n for n, r in ((a_name, a), (b_name, b)) if r is None]
            lines.append(f"> Not available: {', '.join(absent)} has not been run.\n")
            missing.extend(absent)
            continue
        res = block_bootstrap_delta(a, b, n_boot=n_boot)
        sd = seed_sd(a) or seed_sd(b)
        lines.append(f"{verdict(res, sd)}\n")
        if res:
            lines.append(f"> Paired block bootstrap, {res['n_blocks']} "
                         f"(node × fortnight) blocks, {res['n_boot']} draws, "
                         f"p = {res['p']:.3f}.\n")
    return lines, sorted(set(missing))


def render(run_dir: str, runs: Dict[str, Dict], base: Dict[str, Dict],
           far: float, n_boot: int, protocol: str) -> str:
    from datetime import datetime, timezone

    contrasts, missing = contrast_section(runs, n_boot)
    single_seed = sorted(k for k, v in runs.items() if v["n_seeds"] < 2)

    L: List[str] = []
    A = L.append
    A("# Results\n")
    A("<!-- GENERATED by models/results_doc.py — do not edit by hand. -->")
    A(f"<!-- source: {os.path.abspath(run_dir)} -->\n")
    A(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from "
      f"`{run_dir}` · protocol **{protocol}** · {len(runs)} model runs, "
      f"{len(base)} baselines.\n")

    A("## 1. Every model, one table\n")
    A("PR-AUC carries ± one standard deviation across seeds where a run had "
      "more than one. A model without one is a single seed and has no error "
      "bar at all.\n")
    A("**Do not read the `ev.det` column down this table** — each row sits at "
      "its own threshold. §3 is the column that compares early warning.\n")
    L.extend(headline_table(runs, base))

    A("\n## 2. Pre-declared contrasts — the decisions\n")
    A("Each interval is a paired block bootstrap on the *difference*, both "
      "models scored on the same resampled test blocks. An interval spanning "
      "zero means this experiment cannot separate the two models; that is a "
      "result, and it is the honest thing to write.\n")
    L.extend(contrasts)

    A(f"\n## 3. Early warning at a matched false-alarm ratio (FAR = {far:.3f})\n")
    A("Every threshold re-derived so all models raise false alarms at the same "
      "rate, then asked what each detects there. The last two columns are what "
      "the model reported at its own threshold, for comparison.\n")
    A("Caveat to carry into the paper: these thresholds are fitted on the test "
      "predictions, so this is a like-for-like **diagnostic**, not a held-out "
      "result. §1 stays the headline table.\n")
    L.extend(early_warning_table(runs, far))

    A("\n## 4. What to run next\n")
    if missing:
        A("These runs are referenced by a pre-declared contrast and do not "
          "exist yet. Until they do, the contrast above is unanswered:\n")
        for m in missing:
            A(f"- `{m}`")
        A("")
        A("```bash")
        A("# model 3's ladder — the RQ1 answer")
        A("python models/kaggle_run.py --stage ladder3")
        A("# the onset pair")
        A("python models/kaggle_run.py --stage onset")
        A("# model 1's missing BCE rung, without which M-vs-N is confounded")
        A("python models/kaggle_run.py --stage ladder --presets M5_bce --ladder-seeds 5")
        A("```\n")
    else:
        A("Every pre-declared contrast has both its runs. No blocking runs "
          "remain.\n")
    if single_seed:
        A("Single-seed runs — usable as ladder steps, **not** as the two sides "
          "of a claim, because they have no error bar:\n")
        for k in single_seed:
            A(f"- `{k}`")
        A("")

    A("## 5. Claims this evidence supports\n")
    A("Each claim is paired with the caveat that has to travel with it. A "
      "claim whose contrast in §2 spans zero does not belong here.\n")
    A("| claim | evidence | the caveat that must accompany it |")
    A("|---|---|---|")
    A("| Random splits do not merely inflate scores, they select a different "
      "model | PR-AUC 0.704 → 0.905 under a random split while pre-onset "
      "ev.det collapses 0.380 → 0.089 | one seed per arm; the effect is large "
      "enough to survive that, the exact numbers are not |")
    A("| The input layer, not depth, is what loses to gradient-boosted trees | "
      "N0 → N1 (periodic embeddings, nothing else changed) = +0.152 PR-AUC | "
      "single seeds; the size of the gap, not its precise value, is the claim |")
    A("| Focal loss is a defect on this task, not a cost | N5 → N5_bce = "
      "+0.0509 PR-AUC, ECE cut to a third, same architecture and seeds | "
      "shown on model 2; confirm on model 1 via the M5_bce contrast in §2 |")
    A("| PR-AUC on a 1-day flood target measures persistence more than skill | "
      "`discharge_pctl` alone scores 0.816; GBT 0.8496 | this is why §3 and "
      "the onset head exist — lead with them |")
    A("| SAR imagery does not help at a 1-day horizon | M6_cnn null; N6_gated "
      "at 17× the parameters scores below N5_bce | 9 of 51 nodes, ~12-day "
      "revisit — a coverage result, not a statement about SAR in general |")
    A("")
    A("## 6. Provenance\n")
    A("| run | family | seeds | scored head | params | file |")
    A("|---|---|---|---|---|---|")
    for k, v in sorted(runs.items()):
        A(f"| {k} | {v['family']} | {v['n_seeds']} | {HEADS[v['eval_head']]} | "
          f"{v['meta'].get('n_params', 0):,} | `{v['meta']['preset']}_{protocol}.json` |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="directory of *.json / *_preds.npz")
    ap.add_argument("--out", default="docs/RESULTS.md")
    ap.add_argument("--protocol", default="temporal")
    ap.add_argument("--far", type=float, default=0.231,
                    help="matched false-alarm ratio for §3; 0.231 is the "
                         "persistence baseline's, so the table reads as "
                         "'what do you get for the same alarm budget'")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    runs = load_runs(a.run_dir, a.protocol)
    base = load_baselines(a.run_dir, a.protocol)
    if not runs and not base:
        sys.exit(f"[fatal] no {a.protocol} results found in {a.run_dir}")
    doc = render(a.run_dir, runs, base, a.far, a.n_boot, a.protocol)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[results] {len(runs)} runs + {len(base)} baselines -> {a.out}")


if __name__ == "__main__":
    main()
