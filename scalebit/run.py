"""Experiment runner: load model, convert to a format, compute golden outputs, run fault
trials, write everything to CSV. Every step resumes where it stopped.

Results go to  <results_root>/<study>/<model>/<task>/<format>_<dtype>/
    run_info.json        settings, versions, GPU, git commit (one entry per session)
    golden.jsonl         fault-free output per question
    trials_<class>.csv   one row per trial
"""

import csv
import dataclasses
import json
import os
import platform
import subprocess
import time
import zlib

import numpy as np
import torch

from . import __version__
from .formats import make_format
from .tasks import _forward, make_task

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}  # float32: CPU tests only
DTYPE_TAG = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclasses.dataclass
class Config:
    model_id: str
    fmt: str                       # fp | int4_rtn | int4_gptq | mxfp4 | nvfp4
    task: str                      # mmlu | gsm8k
    results_root: str
    study: str = "main"
    dtype: str = "float16"
    gptq_id: str = None
    trials_per_bit: int = 300
    # class -> trials per bit. The NVFP4 tensor scale is 1 value per layer (~0.2% of bits), so a
    # few trials per bit are enough; element bits get more to tighten their upper bound.
    trials_override: dict = dataclasses.field(
        default_factory=lambda: {"element": 1000, "tensor_scale": 30})
    classes: list = None           # None = all classes of the format
    n_questions: int = None        # None = all 100
    max_new_tokens: int = 512
    seed: int = 2026
    restore_check_every: int = 50

    def out_dir(self):
        d = os.path.join(self.results_root, self.study, self.model_id.split("/")[-1], self.task,
                         f"{self.fmt}_{DTYPE_TAG[self.dtype]}")
        os.makedirs(d, exist_ok=True)
        return d

    def n_for(self, cls):
        return int(self.trials_override.get(cls, self.trials_per_bit))


# ---------------------------------------------------------------------------- loading
def load_model(model_id, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = {"dtype": dtype}
    if torch.cuda.is_available():
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert next(model.parameters()).dtype == dtype
    return model, tok


def target_layers(model):
    out = [(n, m) for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in TARGET_MODULES]
    if not out:
        raise RuntimeError("no target linear layers found")
    return out


def git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.out_dir()
        self.dtype = DTYPES[cfg.dtype]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def setup(self):
        cfg = self.cfg
        t0 = time.time()
        self.model, self.tok = load_model(cfg.model_id, self.dtype)
        self.fmt = make_format(cfg.fmt, self.dtype, cfg.gptq_id)
        self.prep_info = self.fmt.prepare(self.model, target_layers(self.model))
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else [])
        self.task = make_task(cfg.task, self.tok, self.device, cfg.n_questions,
                              cfg.max_new_tokens, eos)
        self.classes = cfg.classes or list(self.fmt.classes)
        self.save_constant_bits()
        self.write_run_info(setup_seconds=round(time.time() - t0, 1))
        print(f"{self.fmt.name}: {len(self.fmt.layers)} layers, {self.prep_info}")
        print(f"results: {self.dir}")
        return self

    def write_run_info(self, **extra):
        info = dataclasses.asdict(self.cfg)
        info.update({
            "format_name": self.fmt.name, "scalebit": __version__, "git_commit": git_commit(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "prepare": self.prep_info,
            "fields": {c: sum(self.fmt.field_counts(c)) for c in self.fmt.classes},
            "class_width": self.fmt.classes,
            "metadata_classes": list(self.fmt.metadata_classes),
            "n_weights": sum(l.out_features * l.in_features for l in self.fmt.layers),
        })
        try:
            import transformers
            info["transformers"] = transformers.__version__
        except ImportError:
            pass
        info.update(extra)
        path = os.path.join(self.dir, "run_info.json")
        hist = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    hist = json.load(f)
            except json.JSONDecodeError:
                hist = []
        hist.append(info)
        with open(path, "w") as f:
            json.dump(hist, f, indent=2, default=str)

    def save_constant_bits(self):
        """Bits that never change inside a tensor. Used later to compute the FIT a decoder would
        have if it forced these bits back to their known value (no extra trials needed)."""
        classes = list(self.fmt.classes) if not self.fmt.metadata_classes else \
            list(self.fmt.metadata_classes)
        const = self.fmt.constant_bits(classes)
        path = os.path.join(self.dir, "constant_bits.json")
        with open(path, "w") as f:
            json.dump({c: [{str(b): v for b, v in layer.items()} for layer in layers]
                       for c, layers in const.items()}, f)
        for c, layers in const.items():
            counts = self.fmt.field_counts(c)
            total = sum(counts)
            share = {b: sum(n for n, l in zip(counts, layers) if b in l) / total
                     for b in range(self.fmt.classes[c])}
            fixed = [f"b{b}:{v:.2f}" for b, v in sorted(share.items(), reverse=True) if v > 0]
            print(f"constant bits ({c}, share of fields): {' '.join(fixed) or 'none'}")

    # ------------------------------------------------------------------------ golden
    def golden(self):
        path = os.path.join(self.dir, "golden.jsonl")
        done = {}
        if os.path.exists(path):
            _truncate_partial_line(path)
            with open(path) as f:
                for line in f:
                    g = json.loads(line)
                    done[g["qid"]] = g
        todo = [it for it in self.task.items if it["qid"] not in done]
        with open(path, "a") as f:
            for it in _progress(todo, "golden"):
                g = self.task.golden(self.model, it)
                f.write(json.dumps(g) + "\n")
                f.flush()
                done[g["qid"]] = g
        self.gold = done
        n = len(self.task.items)
        acc = sum(bool(self.gold[it["qid"]]["correct"]) for it in self.task.items)
        bad = sum(not self.gold[it["qid"]]["finite"] for it in self.task.items)
        print(f"golden accuracy {acc}/{n}, non-finite {bad}")
        if self.cfg.task == "gsm8k":
            unstable = sum(not g.get("stable", True) for g in self.gold.values())
            cut = sum(bool(g.get("truncated", False)) for g in self.gold.values())
            print(f"gsm8k: {unstable} unstable and {cut} truncated goldens (both excluded from trials)")
        if bad:
            print("WARNING: NaN/Inf without any fault. This dtype does not suit this model; "
                  "do not run trials.")
        return self.gold

    def usable_qids(self):
        return [it["qid"] for it in self.task.items
                if self.gold[it["qid"]]["finite"] and self.gold[it["qid"]].get("stable", True)
                and not self.gold[it["qid"]].get("truncated", False)]

    # ------------------------------------------------------------------------ checks
    @torch.no_grad()
    def check_determinism(self, n=5):
        bad = 0
        n = min(n, len(self.task.items))
        for it in self.task.items[:n]:
            r = self.task.judge(self.model, it, self.gold[it["qid"]])
            bad += r["outcome"] != "masked"
        print(f"determinism: {bad}/{n} fault-free reruns differ from golden (must be 0)")
        return bad == 0

    @torch.no_grad()
    def check_sanity(self, layer=0, qid=None):
        """A high-impact bit must change the logits; after restore everything is bit-exact.
        The field is the first one where the flip makes the affected weights larger, so the
        test does not depend on whether that bit happened to be 0 or 1."""
        cls, bit = self.fmt.sanity
        lay = self.fmt.layers[layer]
        w = lay.module.weight.data
        field = None
        base = float(w.double().abs().sum())
        for f in range(min(self.fmt.field_counts(cls)[layer], 4096)):
            h = self.fmt.flip(layer, cls, f, bit)
            new = float(w.double().abs().sum())
            self.fmt.restore(h)
            if not np.isfinite(new) or new > base * 1.001:
                field = f
                break
        assert field is not None, "no field found for the sanity flip"
        it = self.task.items[qid if qid is not None else self.usable_qids()[0]]
        w0 = w.clone()
        fp0 = self.fmt.fingerprint()
        ids = it["ids"]
        before = _forward(self.model, ids, 1)[-1].float()
        h = self.fmt.flip(layer, cls, field, bit)
        during = _forward(self.model, ids, 1)[-1].float()
        self.fmt.restore(h)
        after = _forward(self.model, ids, 1)[-1].float()
        exact = torch.equal(w.view(torch.int16), w0.view(torch.int16)) \
            and self.fmt.fingerprint() == fp0
        changed = (not bool(torch.isfinite(during).all())) or \
            float((during - before).abs().max()) > 1e-3
        back = float((after - before).abs().max())
        print(f"sanity: flip {cls} bit {bit} in {lay.name}, field {field}: "
              f"{h['old_value']} -> {h['new_value']}")
        print(f"  output changed: {changed} (must be True); restore bit-exact: {exact}; "
              f"logit diff after restore {back:.2e}")
        del w0
        return changed and exact and back <= 1e-2

    # ------------------------------------------------------------------------ trials
    def plan(self, cls):
        """All trials of one class, ordered round-robin over bit positions so that a partial run
        still covers every position evenly. Each trial has its own seed, so raising the count
        later keeps the earlier trials unchanged."""
        counts = np.array(self.fmt.field_counts(cls), dtype=np.float64)
        p = counts / counts.sum()
        qids = self.usable_qids()
        width = self.fmt.classes[cls]
        base = [self.cfg.seed, zlib.crc32(self.fmt.name.encode()), zlib.crc32(cls.encode()),
                zlib.crc32(self.cfg.task.encode())]
        out = []
        for t in range(self.cfg.n_for(cls)):
            for b in range(width):
                rng = np.random.default_rng(base + [b, t])
                L = int(rng.choice(len(counts), p=p))
                out.append({"bit": b, "t": t, "layer": L,
                            "field": int(rng.integers(0, int(counts[L]))),
                            "qid": qids[int(rng.integers(0, len(qids)))]})
        return out

    FIELDS = ["bit_class", "bit", "word_bit", "bit_role", "t", "layer", "layer_name", "field",
              "qid", "outcome", "golden_pred", "pred", "golden_correct", "faulty_correct",
              "diverge_pos", "old_value", "new_value", "seconds", "status"]

    def run_class(self, cls):
        path = os.path.join(self.dir, f"trials_{cls}.csv")
        done = _done_keys(path)
        todo = [d for d in self.plan(cls) if (d["bit"], d["t"]) not in done]
        print(f"[{self.fmt.name}/{cls}] planned {len(done) + len(todo)}, done {len(done)}, "
              f"left {len(todo)}")
        if not todo:
            return path
        items = {it["qid"]: it for it in self.task.items}
        fp0 = self.fmt.fingerprint()
        with _CSV(path, self.FIELDS) as out:
            for j, d in enumerate(_progress(todo, f"{self.fmt.name}/{cls}")):
                it, gold = items[d["qid"]], self.gold[d["qid"]]
                t0 = time.time()
                h = self.fmt.flip(d["layer"], cls, d["field"], d["bit"])
                try:
                    r = self.task.judge(self.model, it, gold)
                finally:
                    self.fmt.restore(h)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                out.write({
                    "bit_class": cls, "bit": d["bit"], "word_bit": self.fmt.word_bit(cls, d["bit"]),
                    "bit_role": self.fmt.bit_role(cls, d["bit"]), "t": d["t"],
                    "layer": d["layer"], "layer_name": self.fmt.layers[d["layer"]].name,
                    "field": d["field"], "qid": d["qid"], "outcome": r["outcome"],
                    "golden_pred": gold["pred"], "pred": r["pred"],
                    "golden_correct": int(bool(gold["correct"])),
                    "faulty_correct": r["faulty_correct"], "diverge_pos": r["diverge_pos"],
                    "old_value": h["old_value"], "new_value": h["new_value"],
                    "seconds": round(time.time() - t0, 3), "status": "ok"})
                k = self.cfg.restore_check_every
                if k and (j + 1) % k == 0:
                    if self.task.judge(self.model, it, gold)["outcome"] != "masked":
                        raise RuntimeError(f"restore check failed after trial {d}: the fault-free "
                                           "rerun no longer matches golden. Stop and report.")
        if self.fmt.fingerprint() != fp0:
            raise RuntimeError("weights differ from the start of this class: a restore was missed")
        return path

    def run(self):
        for cls in self.classes:
            self.run_class(cls)
        summarize(self.dir)

    PERSIST_FIELDS = ["fault", "bit_class", "bit", "layer", "layer_name", "field", "old_value",
                      "new_value", "questions", "masked", "benign", "sdc", "visible",
                      "golden_correct", "faulty_correct", "seconds", "status"]

    def run_persistent(self, cls, bit, n_faults=20):
        """A memory fault stays until the model is reloaded, so it hits every later question.
        Flip one bit, run every usable question with the fault in place, restore; repeat for
        n_faults random fields. Answers: does one flip break a few answers or the whole model?"""
        path = os.path.join(self.dir, f"persistent_{cls}_b{bit}.csv")
        done = set()
        if os.path.exists(path):
            _truncate_partial_line(path)
            with open(path, newline="") as f:
                done = {int(r["fault"]) for r in csv.DictReader(f) if r.get("status") == "ok"}
        counts = np.array(self.fmt.field_counts(cls), dtype=np.float64)
        qids = self.usable_qids()
        items = {it["qid"]: it for it in self.task.items}
        base = [self.cfg.seed, zlib.crc32(b"persistent"), zlib.crc32(cls.encode()), bit]
        with _CSV(path, self.PERSIST_FIELDS, flush_every=1) as out:
            for i in range(n_faults):
                if i in done:
                    continue
                rng = np.random.default_rng(base + [i])
                L = int(rng.choice(len(counts), p=counts / counts.sum()))
                field = int(rng.integers(0, int(counts[L])))
                t0 = time.time()
                h = self.fmt.flip(L, cls, field, bit)
                tally = {"masked": 0, "benign": 0, "sdc": 0, "visible": 0}
                ok = 0
                try:
                    for q in _progress(qids, f"fault {i}"):
                        r = self.task.judge(self.model, items[q], self.gold[q])
                        tally[r["outcome"]] += 1
                        ok += int(r["faulty_correct"])
                finally:
                    self.fmt.restore(h)
                out.write({"fault": i, "bit_class": cls, "bit": bit, "layer": L,
                           "layer_name": self.fmt.layers[L].name, "field": field,
                           "old_value": h["old_value"], "new_value": h["new_value"],
                           "questions": len(qids), **tally,
                           "golden_correct": sum(bool(self.gold[q]["correct"]) for q in qids),
                           "faulty_correct": ok, "seconds": round(time.time() - t0, 1),
                           "status": "ok"})
                print(f"fault {i}: {h['old_value']} -> {h['new_value']} in "
                      f"{self.fmt.layers[L].name}: {tally}")
        return path


# ---------------------------------------------------------------------------- helpers
def _progress(it, desc):
    try:
        from tqdm.auto import tqdm
        return tqdm(it, desc=desc)
    except ImportError:
        return it


def _truncate_partial_line(path):
    """A session killed mid-write leaves a partial last line; cut it so new rows stay clean."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        f.seek(data.rfind(b"\n") + 1)
        f.truncate()


def _done_keys(path):
    if not os.path.exists(path):
        return set()
    _truncate_partial_line(path)
    with open(path, newline="") as f:
        return {(int(r["bit"]), int(r["t"])) for r in csv.DictReader(f) if r.get("status") == "ok"}


class _CSV:
    def __init__(self, path, fields, flush_every=10):
        self.path, self.fields, self.flush_every = path, fields, flush_every

    def __enter__(self):
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fields)
        if new:
            self.w.writeheader()
        self.n = 0
        return self

    def write(self, row):
        self.w.writerow(row)
        self.n += 1
        if self.n % self.flush_every == 0:
            self.f.flush()
            os.fsync(self.f.fileno())

    def __exit__(self, *exc):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


def summarize(out_dir):
    import pandas as pd
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("trials_") and fn.endswith(".csv"):
            df = pd.read_csv(os.path.join(out_dir, fn))
            for (cls, b), g in df.groupby(["bit_class", "bit"]):
                rows.append({"class": cls, "bit": b, "n": len(g),
                             "sdc": int((g.outcome == "sdc").sum()),
                             "visible": int((g.outcome == "visible").sum()),
                             "benign": int((g.outcome == "benign").sum()),
                             "sec_per_trial": round(g.seconds.mean(), 3)})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
