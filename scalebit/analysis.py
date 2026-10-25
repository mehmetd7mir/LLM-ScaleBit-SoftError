"""Turn trial CSVs into AVF, FIT and protection-cost tables.

Definitions
    AVF of a bit position   share of trials whose answer differs from golden (sdc + visible)
    relative FIT per weight sum over classes and positions of (fields per weight) x AVF;
                            the raw per-bit upset rate is assumed equal for every format
    metadata FIT share      metadata part of that sum
    protection curve        protect bit positions in order of AVF; x = protected bits per weight,
                            y = FIT left
Intervals are 95% Clopper-Pearson. Positions with zero failures still get an upper bound,
which is what the pessimistic ("_hi") columns use.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta

FAIL = ("sdc", "visible")


def clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def load(results_root):
    """One entry per result folder: its last run_info and all its trials."""
    runs = []
    for info_path in sorted(glob.glob(os.path.join(results_root, "**", "run_info.json"),
                                      recursive=True)):
        d = os.path.dirname(info_path)
        with open(info_path) as f:
            info = json.load(f)[-1]
        parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(d, "trials_*.csv")))]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df[df.status == "ok"]
        const = None
        if os.path.exists(os.path.join(d, "constant_bits.json")):
            with open(os.path.join(d, "constant_bits.json")) as f:
                const = json.load(f)
        runs.append({"dir": d, "info": info, "trials": df, "constant_bits": const})
    return runs


def _key(info):
    return {"model": info["model_id"].split("/")[-1], "task": info["task"],
            "format": info["format_name"], "dtype": info["dtype"]}


def environment(runs):
    rows = []
    for r in runs:
        i = r["info"]
        rows.append({**_key(i), "gpu": i["gpu"], "scalebit": i["scalebit"],
                     "commit": i.get("git_commit"), "torch": i["torch"],
                     "transformers": i.get("transformers"), "trials": len(r["trials"])})
    df = pd.DataFrame(rows)
    for col in ("gpu", "scalebit", "commit"):
        if df[col].nunique() > 1:
            print(f"WARNING: results come from different {col} values: {sorted(df[col].unique())}")
    return df


def _protect(df, const):
    """Trials whose flipped bit is constant in its layer count as masked: a decoder that forces
    known-constant bits back to their value corrects exactly these flips."""
    if not const:
        return df
    hit = [str(b) in const.get(c, [{}] * (L + 1))[L] if c in const else False
           for c, L, b in zip(df.bit_class, df.layer, df.bit)]
    df = df.copy()
    df.loc[hit, "outcome"] = "masked"
    return df


def avf_by_position(runs, protected=False):
    rows = []
    for r in runs:
        i, df = r["info"], r["trials"]
        if protected:
            df = _protect(df, r.get("constant_bits"))
        for (cls, b), g in df.groupby(["bit_class", "bit"]):
            n = len(g)
            k = int(g.outcome.isin(FAIL).sum())
            lo, hi = clopper_pearson(k, n)
            rows.append({**_key(i), "class": cls, "bit": int(b), "word_bit": int(g.word_bit.iloc[0]),
                         "role": g.bit_role.iloc[0], "n": n, "fail": k,
                         "sdc": int((g.outcome == "sdc").sum()),
                         "visible": int((g.outcome == "visible").sum()),
                         "benign": int((g.outcome == "benign").sum()),
                         "avf": k / n, "avf_lo": lo, "avf_hi": hi,
                         "fields_per_weight": i["fields"][cls] / i["n_weights"],
                         "metadata": cls in i["metadata_classes"]})
    return pd.DataFrame(rows)


def avf_by_class(pos):
    """Class AVF and its interval from the pooled counts. Every bit position of a class gets the
    same number of trials and holds the same number of bits, so the pooled share equals the
    bit-weighted average, and one Clopper-Pearson interval on the pooled counts is valid
    (and much tighter than adding up per-position bounds)."""
    keys = ["model", "task", "format", "dtype", "class"]
    out = pos.groupby(keys).agg(n=("n", "sum"), fail=("fail", "sum"), sdc=("sdc", "sum"),
                                visible=("visible", "sum"), benign=("benign", "sum"),
                                positions=("bit", "count"),
                                fields_per_weight=("fields_per_weight", "first"),
                                metadata=("metadata", "first")).reset_index()
    ci = [clopper_pearson(int(k), int(n)) for k, n in zip(out.fail, out.n)]
    out["avf"] = out.fail / out.n
    out["avf_lo"] = [c[0] for c in ci]
    out["avf_hi"] = [c[1] for c in ci]
    out["bits_per_weight"] = out.fields_per_weight * out.positions
    out["visible_share"] = np.where(out.fail > 0, out.visible / out.fail.clip(lower=1), np.nan)
    return out


def fit_table(pos):
    cls = avf_by_class(pos)
    cls = cls.assign(fit=cls.bits_per_weight * cls.avf, fit_lo=cls.bits_per_weight * cls.avf_lo,
                     fit_hi=cls.bits_per_weight * cls.avf_hi)
    keys = ["model", "task", "format", "dtype"]
    rows = []
    for k, g in cls.groupby(keys):
        meta, elem = g[g.metadata], g[~g.metadata]
        total = g.fit.sum()
        row = dict(zip(keys, k))
        row.update({
            "bits_per_weight": float(g.bits_per_weight.sum()),
            "fit": total, "fit_lo": g.fit_lo.sum(), "fit_hi": g.fit_hi.sum(),
            "metadata_bit_share": float(meta.bits_per_weight.sum() / g.bits_per_weight.sum()),
            "metadata_fit_share": meta.fit.sum() / total if total > 0 else np.nan,
            # pessimistic for the claim "metadata dominates": elements at their upper bound
            "metadata_fit_share_min": meta.fit.sum() / (meta.fit.sum() + elem.fit_hi.sum())
            if len(meta) and (meta.fit.sum() + elem.fit_hi.sum()) > 0 else np.nan,
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    base = df[df.format.isin(["fp16", "bf16", "fp32"])].set_index(["model", "task", "dtype"])
    df["fit_vs_fp"] = [r.fit / base.fit.get((r.model, r.task, r.dtype), np.nan)
                       for r in df.itertuples()]
    # pessimistic ratio: this format at its upper bound against the baseline at its lower bound
    df["fit_vs_fp_worst"] = [r.fit_hi / base.fit_lo.get((r.model, r.task, r.dtype), np.nan)
                             for r in df.itertuples()]
    return df


def se_amplification(pos):
    """Per-bit AVF of a metadata class over the per-bit AVF of the element class."""
    cls = avf_by_class(pos)
    rows = []
    for k, g in cls.groupby(["model", "task", "format", "dtype"]):
        el = g[g["class"] == "element"]
        if el.empty:
            continue
        e, e_hi = float(el.avf.iloc[0]), float(el.avf_hi.iloc[0])
        for r in g[g.metadata].itertuples():
            rows.append({"model": k[0], "task": k[1], "format": k[2], "dtype": k[3],
                         "metadata": r[5], "se_amp": r.avf / e if e > 0 else np.nan,
                         "se_amp_min": r.avf / e_hi if e_hi > 0 else np.nan})
    return pd.DataFrame(rows)


def protection_curve(pos):
    """Greedy: protect the highest-AVF bit positions first (their FIT per protected bit is the
    AVF itself). Cost = protected stored bits per weight; a concrete scheme (parity, duplication,
    range check) scales this cost, the order stays the same."""
    rows = []
    for k, g in pos.groupby(["model", "task", "format", "dtype"]):
        g = g.assign(fit=g.fields_per_weight * g.avf).sort_values("avf", ascending=False)
        total = g.fit.sum()
        cost, left = 0.0, total
        rows.append({"model": k[0], "task": k[1], "format": k[2], "dtype": k[3], "step": 0,
                     "protected": "", "protected_bits_per_weight": 0.0, "fit_left": total,
                     "fit_left_share": 1.0})
        for step, r in enumerate(g.itertuples(), 1):
            if r.fit <= 0:
                break
            cost += r.fields_per_weight
            left -= r.fit
            rows.append({"model": k[0], "task": k[1], "format": k[2], "dtype": k[3], "step": step,
                         "protected": f"{r[5]}:b{r.bit}", "protected_bits_per_weight": cost,
                         "fit_left": max(left, 0.0),
                         "fit_left_share": max(left, 0.0) / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def persistent_table(results_root):
    """One row per persistent-fault experiment: how many questions one flip breaks."""
    rows = []
    for p in sorted(glob.glob(os.path.join(results_root, "**", "persistent_*.csv"), recursive=True)):
        df = pd.read_csv(p)
        df = df[df.status == "ok"]
        if df.empty:
            continue
        with open(os.path.join(os.path.dirname(p), "run_info.json")) as f:
            info = json.load(f)[-1]
        broken = (df.sdc + df.visible) / df.questions
        rows.append({**_key(info), "class": df.bit_class.iloc[0], "bit": int(df.bit.iloc[0]),
                     "faults": len(df), "mean_share_broken": broken.mean(),
                     "faults_breaking_all": int((broken >= 0.99).sum()),
                     "faults_breaking_none": int((broken == 0).sum()),
                     "mean_accuracy_faulty": (df.faulty_correct / df.questions).mean(),
                     "accuracy_golden": (df.golden_correct / df.questions).mean()})
    return pd.DataFrame(rows)


def report(results_root, out_dir=None, plots=True):
    out_dir = out_dir or os.path.join(results_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    runs = load(results_root)
    if not runs:
        raise FileNotFoundError(f"no results under {results_root}")
    tables = {"environment": environment(runs)}
    pos = avf_by_position(runs)
    tables.update({"avf_by_position": pos, "avf_by_class": avf_by_class(pos),
                   "fit": fit_table(pos), "se_amplification": se_amplification(pos),
                   "protection_curve": protection_curve(pos),
                   "fit_constant_bits_protected": fit_table(avf_by_position(runs, protected=True)),
                   "persistent": persistent_table(results_root)})
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    if plots:
        _plots(pos, tables["protection_curve"], out_dir)
    print(f"tables written to {out_dir}")
    return tables


def _plots(pos, curve, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, g in pos.groupby(["model", "task", "format", "dtype"]):
        classes = list(dict.fromkeys(g["class"]))
        fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3), squeeze=False)
        for ax, cls in zip(axes[0], classes):
            c = g[g["class"] == cls].sort_values("bit")
            err = np.vstack([c.avf - c.avf_lo, c.avf_hi - c.avf])
            ax.bar(c.bit, c.avf, yerr=err, color="#4c72b0", capsize=2)
            ax.set_title(cls)
            ax.set_xlabel("bit position")
            ax.set_ylim(0, 1)
        axes[0][0].set_ylabel("AVF")
        fig.suptitle(" / ".join(k))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "avf_" + "_".join(k) + ".png"), dpi=130)
        plt.close(fig)
    for k, g in curve.groupby(["model", "task", "dtype"]):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for fmt, c in g.groupby("format"):
            ax.step(c.protected_bits_per_weight, c.fit_left_share, where="post", label=fmt)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("protected bits per weight")
        ax.set_ylabel("FIT left (share)")
        ax.legend()
        ax.set_title(" / ".join(k))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "protection_" + "_".join(k) + ".png"), dpi=130)
        plt.close(fig)
