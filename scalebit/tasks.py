"""Evaluation tasks. Each task gives a golden (fault-free) reference per question and judges a
faulty run against it.

mmlu   tinyMMLU, 100 questions, 5-shot prompt ending in "Answer:". One forward pass; the
       prediction is the argmax over the logits of " A", " B", " C", " D".
gsm8k  tinyGSM8K, 100 questions, 5-shot prompt, greedy generation. A faulty run is first
       checked with a single teacher-forced pass over prompt + golden tokens: if the argmax at
       every position equals the golden token, greedy decoding would produce exactly the golden
       text, so the fault is masked. Only when the tokens diverge do we generate from the
       divergence point and compare the final numeric answer.

Outcomes: masked (identical output), benign (gsm8k only: text changed, same answer),
sdc (answer changed silently), visible (NaN/Inf in the logits).
"""

import re

import torch

MMLU_ID = ("tinyBenchmarks/tinyMMLU", "all", "test")
GSM8K_ID = ("tinyBenchmarks/tinyGSM8K", "main", "test")


def _forward(model, ids, keep):
    try:
        return model(ids, use_cache=False, logits_to_keep=keep).logits[0]
    except TypeError:  # older transformers
        return model(ids, use_cache=False).logits[0, -keep:]


class MMLU:
    name = "mmlu"

    def __init__(self, tok, device, n_questions=None):
        from datasets import load_dataset
        ds = load_dataset(*MMLU_ID[:2], split=MMLU_ID[2])
        letters = []
        for L in "ABCD":
            ids = tok.encode(" " + L, add_special_tokens=False)
            assert len(ids) == 1, (L, ids)
            letters.append(ids[0])
        self.letters = torch.tensor(letters, device=device)
        self.items = []
        for i, r in enumerate(ds):
            if n_questions is not None and i >= n_questions:
                break
            assert r["input_formatted"].endswith("Answer:")
            ids = tok.encode(r["input_formatted"])  # adds BOS where the model uses one (Llama)
            self.items.append({"qid": i, "ids": torch.tensor([ids], device=device),
                               "answer": int(r["answer"])})

    @torch.no_grad()
    def _predict(self, model, item):
        logits = _forward(model, item["ids"], 1)[-1].float()
        if not bool(torch.isfinite(logits).all()):
            return None, None
        l4 = logits[self.letters]
        top2 = l4.topk(2).values
        return int(l4.argmax()), float(top2[0] - top2[1])

    def golden(self, model, item):
        pred, margin = self._predict(model, item)
        return {"qid": item["qid"], "pred": pred, "finite": pred is not None,
                "correct": pred == item["answer"], "margin": margin}

    def judge(self, model, item, gold):
        pred, _ = self._predict(model, item)
        if pred is None:
            outcome = "visible"
        elif pred == gold["pred"]:
            outcome = "masked"
        else:
            outcome = "sdc"
        return {"outcome": outcome, "pred": "" if pred is None else pred,
                "faulty_correct": int(pred == item["answer"]), "diverge_pos": ""}


_NUM = re.compile(r"-?[\d,]*\.?\d+")
_STRICT = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_DONE = re.compile(r"####[^\n]*\d[^\n]*\n")


def extract_answer(text):
    m = _STRICT.search(text)
    if m:
        s = m.group(1)
    else:
        nums = _NUM.findall(text.split("Question:")[0])
        if not nums:
            return ""
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        x = float(s)
        return str(int(x)) if x == int(x) else str(x)
    except ValueError:
        return s


class GSM8K:
    name = "gsm8k"

    def __init__(self, tok, device, n_questions=None, max_new_tokens=512, eos_ids=()):
        from datasets import load_dataset
        ds = load_dataset(*GSM8K_ID[:2], split=GSM8K_ID[2])
        self.tok = tok
        self.max_new = max_new_tokens
        self.eos = set(eos_ids) | ({tok.eos_token_id} if tok.eos_token_id is not None else set())
        self.items = []
        for i, r in enumerate(ds):
            if n_questions is not None and i >= n_questions:
                break
            ids = tok.encode(r["input_formatted"])  # adds BOS where the model uses one (Llama)
            self.items.append({"qid": i, "ids": torch.tensor([ids], device=device),
                               "answer": extract_answer(r["answer"])})

    def _stop(self, tokens):
        if tokens and tokens[-1] in self.eos:
            return True
        text = self.tok.decode(tokens, skip_special_tokens=True)
        return bool(_DONE.search(text)) or "Question:" in text

    @torch.no_grad()
    def _generate(self, model, ids, prefix):
        """Greedy decoding with KV cache, starting after `prefix` (list of already fixed tokens).
        Returns (tokens, finite)."""
        tokens = list(prefix)
        if tokens and self._stop(tokens):
            return tokens, True
        full = torch.cat([ids, torch.tensor([tokens], dtype=ids.dtype, device=ids.device)], 1) \
            if tokens else ids
        out = model(full, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[0, -1]
        while len(tokens) < self.max_new:
            if not bool(torch.isfinite(logits).all()):
                return tokens, False
            t = int(logits.argmax())
            tokens.append(t)
            if self._stop(tokens):
                break
            out = model(torch.tensor([[t]], device=ids.device), past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
        return tokens, True

    @torch.no_grad()
    def _teacher_forced(self, model, ids, tokens):
        """Argmax at every golden position in one pass. Returns (argmax list, finite)."""
        if not tokens:
            return [], True
        full = torch.cat([ids, torch.tensor([tokens], dtype=ids.dtype, device=ids.device)], 1)
        logits = _forward(model, full, len(tokens) + 1)[:-1]
        if not bool(torch.isfinite(logits).all()):
            return None, False
        return logits.argmax(-1).tolist(), True

    def golden(self, model, item, max_fix=8):
        tokens, finite = self._generate(model, item["ids"], [])
        # Cached decoding and a full forward pass can disagree on near-ties. The golden sequence
        # is made consistent with the teacher-forced check, so a fault-free run is always masked.
        fixes = 0
        while finite and fixes < max_fix:
            am, finite = self._teacher_forced(model, item["ids"], tokens)
            if not finite:
                break
            p = next((k for k, (a, b) in enumerate(zip(am, tokens)) if a != b), None)
            if p is None:
                break
            fixes += 1
            tokens, finite = self._generate(model, item["ids"], tokens[:p] + [am[p]])
        text = self.tok.decode(tokens, skip_special_tokens=True)
        pred = extract_answer(text)
        # hit the token limit before finishing: the "answer" would be a number cut mid-reasoning
        truncated = len(tokens) >= self.max_new and not self._stop(tokens)
        return {"qid": item["qid"], "tokens": tokens, "pred": pred, "finite": finite,
                "correct": pred == item["answer"], "fixes": fixes, "truncated": truncated,
                "stable": finite and fixes < max_fix}

    def judge(self, model, item, gold):
        tokens = gold["tokens"]
        am, finite = self._teacher_forced(model, item["ids"], tokens)
        if not finite:
            return {"outcome": "visible", "pred": "", "faulty_correct": 0, "diverge_pos": ""}
        p = next((k for k, (a, b) in enumerate(zip(am, tokens)) if a != b), None)
        if p is None:
            return {"outcome": "masked", "pred": gold["pred"],
                    "faulty_correct": int(gold["correct"]), "diverge_pos": ""}
        new, finite = self._generate(model, item["ids"], tokens[:p] + [am[p]])
        if not finite:
            return {"outcome": "visible", "pred": "", "faulty_correct": 0, "diverge_pos": p}
        pred = extract_answer(self.tok.decode(new, skip_special_tokens=True))
        outcome = "benign" if pred == gold["pred"] else "sdc"
        return {"outcome": outcome, "pred": pred, "faulty_correct": int(pred == item["answer"]),
                "diverge_pos": p}


def make_task(name, tok, device, n_questions=None, max_new_tokens=512, eos_ids=()):
    if name == "mmlu":
        return MMLU(tok, device, n_questions)
    if name == "gsm8k":
        return GSM8K(tok, device, n_questions, max_new_tokens, eos_ids)
    raise ValueError(name)
