"""INT4 with group-wise FP16 scale and 4-bit zero point, in the AutoGPTQ v1 ("gptq") layout.

Two ways to fill it:
    int4_rtn   our round-to-nearest, symmetric, group 128 (same rule AutoGPTQ uses for its
               scale and zero point: scale = max|w| / 7.5, zero = 8)
    int4_gptq  a published GPTQ checkpoint (e.g. Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)
Both end up in exactly the same stored arrays, so the fault model is identical.

Stored arrays per layer (CPU):
    qweight int32   [in/8, out]    8 elements per word along the input dim, element j at bits 4j..4j+3
    scales  float16 [G, out]
    qzeros  int32   [G, out/8]     8 zero points per word along the output dim
    g_idx   int64   [in]           group of each input row (sorted unless desc_act was used)
Dequantization: W[o, i] = scale[g, o] * (q[i, o] - (z[g, o] + 1)),  g = g_idx[i].
The "+1" is how the v1 format stores zero points (a symmetric 4-bit zero of 8 is stored as 7).
We dequantize in plain PyTorch on purpose: fused kernels (ExLlama, Marlin) repack the weights
into their own copy, and a flip in our arrays would then not reach the computation.
"""

import json
import os

import torch

from ..bits import float_role, xor_bit
from .base import Format, Layer

SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)


def unpack_qweight(qweight):
    q = (qweight.unsqueeze(1) >> SHIFTS.to(qweight.device).view(1, 8, 1)) & 0xF
    return q.reshape(-1, qweight.shape[1])


def unpack_qzeros(qzeros):
    z = (qzeros.unsqueeze(2) >> SHIFTS.to(qzeros.device).view(1, 1, 8)) & 0xF
    return z.reshape(qzeros.shape[0], -1)


def _pack_rows(q):
    """[in, out] values 0..15 -> [in/8, out] int32."""
    q = q.to(torch.int64)
    w = torch.zeros(q.shape[0] // 8, q.shape[1], dtype=torch.int64, device=q.device)
    for j in range(8):
        w |= q[j::8] << (4 * j)
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)


def _pack_cols(z):
    """[G, out] values 0..15 -> [G, out/8] int32."""
    z = z.to(torch.int64)
    w = torch.zeros(z.shape[0], z.shape[1] // 8, dtype=torch.int64, device=z.device)
    for j in range(8):
        w |= z[:, j::8] << (4 * j)
    return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)


def dequantize(qweight, scales, qzeros, g_idx, zero_offset, dtype):
    """Whole layer -> [out, in] in dtype."""
    q = unpack_qweight(qweight)
    z = unpack_qzeros(qzeros) + zero_offset
    w_t = scales.float()[g_idx] * (q - z[g_idx]).float()
    return w_t.t().contiguous().to(dtype)


def quantize_rtn(w, group_size=128):
    """Symmetric RTN, AutoGPTQ conventions. w: [out, in] -> (qweight, scales, qzeros, g_idx)."""
    out_f, in_f = w.shape
    assert in_f % group_size == 0 and out_f % 8 == 0
    v = w.float().t().reshape(in_f // group_size, group_size, out_f)     # [G, gs, out]
    xmax = v.abs().amax(1)                                              # [G, out]
    scale = torch.where(xmax > 0, xmax / 7.5, torch.ones_like(xmax))
    scale = scale.half()
    q = torch.clamp(torch.round(v / scale.float().unsqueeze(1)) + 8, 0, 15)
    q = q.reshape(in_f, out_f)
    zeros = torch.full((in_f // group_size, out_f), 7, dtype=torch.int64, device=w.device)
    g_idx = torch.arange(in_f, device=w.device) // group_size
    return _pack_rows(q), scale, _pack_cols(zeros), g_idx


class INT4(Format):
    classes = {"element": 4, "scale": 16, "zero": 4}
    metadata_classes = ("scale", "zero")
    sanity = ("scale", 14)

    def __init__(self, dtype, source="rtn", gptq_id=None, group_size=128):
        super().__init__(dtype)
        assert source in ("rtn", "gptq")
        self.source, self.gptq_id = source, gptq_id
        self.group_size = group_size
        self.zero_offset = 1
        self.name = f"int4_{source}"

    # -- preparation --------------------------------------------------------------
    @torch.no_grad()
    def prepare(self, model, targets):
        if self.source == "gptq":
            reader = _CheckpointReader(self.gptq_id)
            qc = reader.quant_config
            assert qc["bits"] == 4, qc
            self.group_size = qc["group_size"]
            self.zero_offset = 0 if qc.get("checkpoint_format", "gptq") == "gptq_v2" else 1
        errs = []
        for name, mod in targets:
            w = mod.weight.data
            dev = w.device
            if self.source == "rtn":
                qweight, scales, qzeros, g_idx = quantize_rtn(w, self.group_size)
            else:
                qweight = reader.get(name + ".qweight", dev)
                scales = reader.get(name + ".scales", dev).half()
                qzeros = reader.get(name + ".qzeros", dev)
                g_idx = (reader.get(name + ".g_idx", dev).long() if reader.has(name + ".g_idx")
                         else torch.arange(w.shape[1], device=dev) // self.group_size)
                if mod.bias is not None and reader.has(name + ".bias"):
                    mod.bias.data.copy_(reader.get(name + ".bias", dev).to(mod.bias.dtype))
            deq = dequantize(qweight, scales, qzeros, g_idx, self.zero_offset, self.dtype)
            assert deq.shape == w.shape, (name, deq.shape, w.shape)
            errs.append(float((deq.float() - w.float()).norm() / w.float().norm()))
            w.copy_(deq)
            layer = Layer(name, mod)
            layer.store = {"qweight": qweight.cpu(), "scales": scales.cpu(),
                           "qzeros": qzeros.cpu(), "g_idx": g_idx.cpu()}
            layer.n_groups = scales.shape[0]
            g_cpu = layer.store["g_idx"]
            sorted_ = bool(torch.equal(g_cpu, torch.arange(len(g_cpu)) // self.group_size))
            layer.group_rows = None if sorted_ else \
                [torch.nonzero(g_cpu == g).flatten() for g in range(layer.n_groups)]
            self.layers.append(layer)
            del deq
        info = {"rel_error_mean": sum(errs) / len(errs), "rel_error_max": max(errs),
                "group_size": self.group_size, "zero_offset": self.zero_offset}
        if self.source == "gptq":
            info["copied_other_tensors"] = reader.copy_unquantized(model)
            info["desc_act"] = bool(qc.get("desc_act", False))
        return info

    # -- fields -------------------------------------------------------------------
    def field_counts(self, cls):
        if cls == "element":
            return [l.out_features * l.in_features for l in self.layers]
        return [l.out_features * l.n_groups for l in self.layers]

    def bit_role(self, cls, bit):
        if cls == "scale":
            return float_role(bit, 10, 16)
        return "int"

    def _rows(self, layer, g):
        if layer.group_rows is None:
            return slice(g * self.group_size, (g + 1) * self.group_size)
        return layer.group_rows[g]

    def _apply(self, layer, cls, field, bit):
        s = layer.store
        if cls == "element":
            o, i = divmod(field, layer.in_features)
            pos = (i // 8, o)
            sh = 4 * (i % 8) + bit
            old = (int(s["qweight"][pos]) >> (4 * (i % 8))) & 0xF
            xor_bit(s["qweight"], pos, sh)
            return [(s["qweight"], pos, sh)], (o, slice(i, i + 1)), str(old), str(old ^ (1 << bit))
        g, o = divmod(field, layer.out_features)
        region = (o, self._rows(layer, g))
        if cls == "scale":
            old_v = float(s["scales"][g, o])
            xor_bit(s["scales"], (g, o), bit)
            return [(s["scales"], (g, o), bit)], region, f"{old_v:.6g}", \
                f"{float(s['scales'][g, o]):.6g}"
        pos = (g, o // 8)
        sh = 4 * (o % 8) + bit
        old = (int(s["qzeros"][pos]) >> (4 * (o % 8))) & 0xF
        xor_bit(s["qzeros"], pos, sh)
        return [(s["qzeros"], pos, sh)], region, str(old), str(old ^ (1 << bit))

    def _recompute(self, layer, region):
        o, rows = region
        s = layer.store
        idx = torch.arange(layer.in_features)[rows]
        q = (s["qweight"][idx // 8, o] >> (4 * (idx % 8)).to(torch.int32)) & 0xF
        g = s["g_idx"][idx]
        z = ((s["qzeros"][g, o // 8] >> (4 * (o % 8))) & 0xF) + self.zero_offset
        return (s["scales"][g, o].float() * (q - z).float()).to(self.dtype)


class _CheckpointReader:
    def __init__(self, repo_id):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        self._safe_open = safe_open
        path = snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors"])
        with open(os.path.join(path, "config.json")) as f:
            self.quant_config = json.load(f)["quantization_config"]
        self.where = {}
        for fn in os.listdir(path):
            if fn.endswith(".safetensors"):
                with safe_open(os.path.join(path, fn), framework="pt") as f:
                    for k in f.keys():
                        self.where[k] = os.path.join(path, fn)

    def has(self, key):
        return key in self.where

    def get(self, key, device):
        with self._safe_open(self.where[key], framework="pt") as f:
            return f.get_tensor(key).to(device)

    @torch.no_grad()
    def copy_unquantized(self, model):
        """Embeddings, norms and lm_head in the checkpoint may differ slightly from the base
        model; use the checkpoint's copy so the GPTQ model is exactly the published one."""
        sd = model.state_dict()
        skip = (".qweight", ".scales", ".qzeros", ".g_idx")
        n = 0
        for k in self.where:
            if k in sd and not k.endswith(skip):
                t = self.get(k, sd[k].device).to(sd[k].dtype)
                if not torch.equal(t, sd[k]):
                    sd[k].copy_(t)
                    n += 1
        return n
