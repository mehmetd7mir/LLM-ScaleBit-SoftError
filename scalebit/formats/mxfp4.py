"""MXFP4 (OCP Microscaling Formats v1.0).

Blocks of 32 weights along the input dimension share one E8M0 scale (8 exponent bits,
value 2^(byte-127), 0xFF = NaN for the whole block). Elements are E2M1.
Quantization follows the OCP conversion rule (spec p. 15):
    shared_exp = floor(log2(amax)) - 2,   element = RNE_E2M1(v / 2^shared_exp), saturate at 6.

Stored arrays per layer (CPU):
    codes  uint8 [out, in/2]   two elements per byte, even index in the low nibble
    scales uint8 [out, in/32]
"""

import torch

from ..bits import (e2m1_encode, e2m1_role, e2m1_table, pack_nibbles, unpack_nibbles,
                    xor_bit)
from .base import Format, Layer

BLOCK = 32
E2M1_EMAX = 2
BIAS = 127
NAN_SCALE = 255


def quantize(w):
    out_f, in_f = w.shape
    assert in_f % BLOCK == 0, in_f
    v = w.float().reshape(out_f, in_f // BLOCK, BLOCK)
    amax = v.abs().amax(-1)
    exp = torch.where(amax > 0, torch.floor(torch.log2(amax)), torch.full_like(amax, -BIAS))
    exp = (exp - E2M1_EMAX).clamp(-BIAS, BIAS)
    scaled = torch.ldexp(v, (-exp).to(torch.int32).unsqueeze(-1))
    codes = e2m1_encode(scaled).reshape(out_f, in_f)
    return codes, (exp + BIAS).to(torch.uint8)


def dequantize(codes, scales, dtype):
    """codes [..., k*32] unpacked uint8, scales [..., k] uint8 -> values in dtype."""
    v = e2m1_table(codes.device)[codes.long()]
    v = v.reshape(*codes.shape[:-1], -1, BLOCK)
    v = torch.ldexp(v, (scales.to(torch.int32) - BIAS).unsqueeze(-1))
    v = torch.where((scales == NAN_SCALE).unsqueeze(-1), torch.full_like(v, float("nan")), v)
    return v.reshape(codes.shape).to(dtype)


class MXFP4(Format):
    name = "mxfp4"
    classes = {"element": 4, "scale": 8}
    metadata_classes = ("scale",)
    sanity = ("scale", 7)

    @torch.no_grad()
    def prepare(self, model, targets):
        errs = []
        for name, mod in targets:
            w = mod.weight.data
            codes, scales = quantize(w)
            deq = dequantize(codes, scales, self.dtype)
            errs.append(float((deq.float() - w.float()).norm() / w.float().norm()))
            w.copy_(deq)
            layer = Layer(name, mod)
            layer.store = {"codes": pack_nibbles(codes).cpu(), "scales": scales.cpu()}
            self.layers.append(layer)
            del codes, scales, deq
        return {"rel_error_mean": sum(errs) / len(errs), "rel_error_max": max(errs)}

    def field_counts(self, cls):
        if cls == "element":
            return [l.out_features * l.in_features for l in self.layers]
        return [l.out_features * (l.in_features // BLOCK) for l in self.layers]

    def stored_words(self, layer, cls):
        if cls == "element":
            return unpack_nibbles(layer.store["codes"]).to(torch.int16)
        return layer.store["scales"].to(torch.int16)

    def bit_role(self, cls, bit):
        return e2m1_role(bit) if cls == "element" else "exponent"

    def _apply(self, layer, cls, field, bit):
        s = layer.store
        if cls == "element":
            o, i = divmod(field, layer.in_features)
            blk = i // BLOCK
            old = int(unpack_nibbles(s["codes"][o:o + 1, i // 2:i // 2 + 1])[0, i % 2])
            xor_bit(s["codes"], (o, i // 2), bit + 4 * (i % 2))
            new = old ^ (1 << bit)
            undo = [(s["codes"], (o, i // 2), bit + 4 * (i % 2))]
            old_v, new_v = f"e2m1:{old}", f"e2m1:{new}"
        else:
            o, blk = divmod(field, layer.in_features // BLOCK)
            old, new = xor_bit(s["scales"], (o, blk), bit)
            undo = [(s["scales"], (o, blk), bit)]
            old_v = f"2^{old - BIAS}"
            new_v = "NaN" if new == NAN_SCALE else f"2^{new - BIAS}"
        return undo, (o, slice(blk * BLOCK, (blk + 1) * BLOCK)), old_v, new_v

    def _recompute(self, layer, region):
        o, cols = region
        blk = cols.start // BLOCK
        codes = unpack_nibbles(layer.store["codes"][o:o + 1, cols.start // 2:cols.stop // 2])
        sc = layer.store["scales"][o:o + 1, blk:blk + 1]
        return dequantize(codes, sc, self.dtype)[0]
