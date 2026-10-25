"""NVFP4 (NVIDIA): E2M1 elements, one E4M3 scale per 16 weights along the input dimension,
and one FP32 decode scale per tensor. Quantization follows the NVFP4 report, Eq. 1-4 (p. 16):
    s_enc   = 6 * 448 / amax(tensor)            s_dec = 1 / s_enc   (stored, FP32)
    scale_b = e4m3(amax(block) / 6 * s_enc)     (stored, round to nearest even)
    element = RNE_E2M1(v / (scale_b * s_dec)),  saturate at 6
    value   = element * scale_b * s_dec

Stored arrays per layer (CPU):
    codes        uint8   [out, in/2]
    scales       uint8   [out, in/16]   raw E4M3 bits (0x7F / 0xFF = NaN)
    tensor_scale float32 [1]            s_dec
"""

import struct

import torch

from ..bits import (E4M3_MAX, e2m1_encode, e2m1_role, e2m1_table, e4m3_decode, e4m3_encode,
                    e4m3_role, float_role, pack_nibbles, unpack_nibbles, xor_bit)
from .base import Format, Layer

BLOCK = 16
E2M1_MAX = 6.0


def quantize(w):
    out_f, in_f = w.shape
    assert in_f % BLOCK == 0, in_f
    v = w.float().reshape(out_f, in_f // BLOCK, BLOCK)
    amax_t = v.abs().max()
    s_enc = (E2M1_MAX * E4M3_MAX / amax_t) if amax_t > 0 else torch.tensor(1.0, device=w.device)
    s_dec = (1.0 / s_enc).reshape(1).float()
    scales = e4m3_encode(v.abs().amax(-1) / E2M1_MAX * s_enc)
    block_dec = e4m3_decode(scales) * s_dec
    enc = torch.where(block_dec > 0, 1.0 / block_dec, torch.zeros_like(block_dec))
    codes = e2m1_encode(v * enc.unsqueeze(-1)).reshape(out_f, in_f)
    return codes, scales, s_dec


def dequantize(codes, scales, s_dec, dtype):
    v = e2m1_table(codes.device)[codes.long()].reshape(*codes.shape[:-1], -1, BLOCK)
    v = v * (e4m3_decode(scales) * s_dec.to(codes.device)).unsqueeze(-1)
    return v.reshape(codes.shape).to(dtype)


def _f32(x):
    return struct.unpack("<f", struct.pack("<I", x & 0xFFFFFFFF))[0]


class NVFP4(Format):
    name = "nvfp4"
    classes = {"element": 4, "scale": 8, "tensor_scale": 32}
    metadata_classes = ("scale", "tensor_scale")
    sanity = ("tensor_scale", 30)  # block scales mostly have exponent MSB = 1 already

    @torch.no_grad()
    def prepare(self, model, targets):
        errs, n_max_scale = [], 0
        for name, mod in targets:
            w = mod.weight.data
            codes, scales, s_dec = quantize(w)
            deq = dequantize(codes, scales, s_dec, self.dtype)
            errs.append(float((deq.float() - w.float()).norm() / w.float().norm()))
            n_max_scale += int((scales == 0x7E).sum())
            w.copy_(deq)
            layer = Layer(name, mod)
            layer.store = {"codes": pack_nibbles(codes).cpu(), "scales": scales.cpu(),
                           "tensor_scale": s_dec.cpu()}
            self.layers.append(layer)
            del codes, scales, deq
        return {"rel_error_mean": sum(errs) / len(errs), "rel_error_max": max(errs),
                "blocks_with_scale_448": n_max_scale}

    def field_counts(self, cls):
        if cls == "element":
            return [l.out_features * l.in_features for l in self.layers]
        if cls == "scale":
            return [l.out_features * (l.in_features // BLOCK) for l in self.layers]
        return [1 for _ in self.layers]

    def stored_words(self, layer, cls):
        if cls == "element":
            return unpack_nibbles(layer.store["codes"]).to(torch.int16)
        if cls == "scale":
            return layer.store["scales"].to(torch.int16)
        return layer.store["tensor_scale"].view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    def bit_role(self, cls, bit):
        if cls == "element":
            return e2m1_role(bit)
        if cls == "scale":
            return e4m3_role(bit)
        return float_role(bit, 23, 32)

    def _apply(self, layer, cls, field, bit):
        s = layer.store
        if cls == "element":
            o, i = divmod(field, layer.in_features)
            old = int(unpack_nibbles(s["codes"][o:o + 1, i // 2:i // 2 + 1])[0, i % 2])
            xor_bit(s["codes"], (o, i // 2), bit + 4 * (i % 2))
            undo = [(s["codes"], (o, i // 2), bit + 4 * (i % 2))]
            blk = i // BLOCK
            return undo, (o, slice(blk * BLOCK, (blk + 1) * BLOCK)), \
                f"e2m1:{old}", f"e2m1:{old ^ (1 << bit)}"
        if cls == "scale":
            o, blk = divmod(field, layer.in_features // BLOCK)
            old, new = xor_bit(s["scales"], (o, blk), bit)
            dec = e4m3_decode(torch.tensor([old, new], dtype=torch.uint8)).tolist()
            return [(s["scales"], (o, blk), bit)], (o, slice(blk * BLOCK, (blk + 1) * BLOCK)), \
                f"{dec[0]:.6g}", f"{dec[1]:.6g}"
        old, new = xor_bit(s["tensor_scale"], (0,), bit)
        return [(s["tensor_scale"], (0,), bit)], (None, None), \
            f"{_f32(old):.6g}", f"{_f32(new):.6g}"

    def _recompute(self, layer, region):
        s = layer.store
        o, cols = region
        if o is None:  # tensor scale: whole layer, done on the GPU
            dev = layer.module.weight.device
            return dequantize(unpack_nibbles(s["codes"].to(dev)), s["scales"].to(dev),
                              s["tensor_scale"].to(dev), self.dtype)
        blk = cols.start // BLOCK
        codes = unpack_nibbles(s["codes"][o:o + 1, cols.start // 2:cols.stop // 2])
        return dequantize(codes, s["scales"][o:o + 1, blk:blk + 1], s["tensor_scale"],
                          self.dtype)[0]
