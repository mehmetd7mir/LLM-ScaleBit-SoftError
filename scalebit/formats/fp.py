"""Unquantized baseline: weights stored in the compute dtype (FP16 or BF16)."""

import torch

from ..bits import float_role, xor_bit
from .base import Format, Layer, fmt_value

LAYOUT = {  # dtype -> ({class: (first bit, width)}, mantissa bits, word bits, name)
    torch.float16: ({"sign": (15, 1), "exponent": (10, 5), "mantissa": (0, 10)}, 10, 16, "fp16"),
    torch.bfloat16: ({"sign": (15, 1), "exponent": (7, 8), "mantissa": (0, 7)}, 7, 16, "bf16"),
    # float32 is only for quick CPU tests (FP16 matmuls are very slow on CPU)
    torch.float32: ({"sign": (31, 1), "exponent": (23, 8), "mantissa": (0, 23)}, 23, 32, "fp32"),
}


class FP(Format):
    def __init__(self, dtype):
        super().__init__(dtype)
        if dtype not in LAYOUT:
            raise ValueError(f"unsupported dtype {dtype}")
        self.layout, self.mant_bits, self.word_bits, self.name = LAYOUT[dtype]
        self.classes = {c: w for c, (_, w) in self.layout.items()}
        self.sanity = ("exponent", self.classes["exponent"] - 1)

    def prepare(self, model, targets):
        for name, mod in targets:
            assert mod.weight.dtype == self.dtype, (name, mod.weight.dtype)
            self.layers.append(Layer(name, mod))
        return {}

    def field_counts(self, cls):
        return [l.out_features * l.in_features for l in self.layers]

    def word_bit(self, cls, bit):
        return self.layout[cls][0] + bit

    def bit_role(self, cls, bit):
        return float_role(self.word_bit(cls, bit), self.mant_bits, self.word_bits)

    @torch.no_grad()
    def flip(self, L, cls, field, bit):
        layer = self.layers[L]
        o, i = divmod(field, layer.in_features)
        w = layer.module.weight.data
        old_v = float(w[o, i])
        xor_bit(w, (o, i), self.word_bit(cls, bit))
        return {"L": L, "o": o, "i": i, "bit": self.word_bit(cls, bit),
                "old_value": fmt_value(old_v), "new_value": fmt_value(float(w[o, i]))}

    @torch.no_grad()
    def restore(self, h):
        xor_bit(self.layers[h["L"]].module.weight.data, (h["o"], h["i"]), h["bit"])
