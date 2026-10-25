"""Common machinery for all formats.

A format keeps the stored representation of each target layer (codes, scales, ...) on the
CPU and the dequantized weight inside the model on the GPU. A fault flips one bit of the
stored representation, recomputes only the weights that depend on it, and writes them into
the model. Restoring flips the bit back and copies the saved weights back, so the model is
bit-exact again after every trial.

Field indices are row-major over the stored array of each class, e.g. for a per-block scale
`field = row * n_blocks + block`.
"""

import zlib

import torch

from ..bits import xor_bit


class Layer:
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.out_features, self.in_features = module.weight.shape
        self.store = {}


class Format:
    name = None
    classes = {}           # bit class -> width in bits
    metadata_classes = ()  # classes that are not weight elements
    sanity = None          # (class, bit) that must visibly change the output when flipped

    def __init__(self, dtype):
        self.dtype = dtype
        self.layers = []

    # -- to implement -------------------------------------------------------------
    def prepare(self, model, targets):
        raise NotImplementedError

    def field_counts(self, cls):
        raise NotImplementedError

    def bit_role(self, cls, bit):
        raise NotImplementedError

    def _apply(self, layer, cls, field, bit):
        """Flip the stored bit. Return (undo list, weight region, old value, new value)."""
        raise NotImplementedError

    def _recompute(self, layer, region):
        """Dequantize `region` from the stored representation (values in self.dtype, CPU or GPU)."""
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------
    def word_bit(self, cls, bit):
        return bit

    def bits_per_weight(self):
        n_weights = sum(l.out_features * l.in_features for l in self.layers)
        return {c: sum(self.field_counts(c)) * w / n_weights for c, w in self.classes.items()}

    def total_bits(self, cls):
        return sum(self.field_counts(cls)) * self.classes[cls]

    @staticmethod
    def _get(w, region):
        o, cols = region
        if o is None:
            return w
        return w[o, cols]

    @staticmethod
    def _set(w, region, values):
        o, cols = region
        if o is None:
            w.copy_(values)
        else:
            w[o, cols] = values

    @torch.no_grad()
    def flip(self, L, cls, field, bit):
        layer = self.layers[L]
        undo, region, old_v, new_v = self._apply(layer, cls, field, bit)
        w = layer.module.weight.data
        saved = self._get(w, region).clone()
        values = self._recompute(layer, region)
        self._set(w, region, values.to(device=w.device, dtype=w.dtype))
        return {"L": L, "undo": undo, "region": region, "saved": saved,
                "old_value": old_v, "new_value": new_v}

    @torch.no_grad()
    def restore(self, h):
        layer = self.layers[h["L"]]
        for t, idx, bit in reversed(h["undo"]):
            xor_bit(t, idx, bit)
        self._set(layer.module.weight.data, h["region"], h["saved"])

    def stored_words(self, layer, cls):
        """Raw stored values of one class in one layer, as integers (for bit statistics)."""
        raise NotImplementedError

    @torch.no_grad()
    def constant_bits(self, classes=None):
        """For every layer and class: the bit positions that hold the same value in all fields,
        and that value. A decoder that forces these bits back to their known value corrects any
        flip in them at no memory cost (one small pattern per tensor)."""
        out = {}
        for cls in classes or self.metadata_classes:
            width = self.classes[cls]
            per_layer = []
            for layer in self.layers:
                w = self.stored_words(layer, cls)
                const = {}
                if w.numel() <= 1:  # a single value has nothing to compare against
                    per_layer.append(const)
                    continue
                for b in range(width):
                    bits = (w >> b) & 1
                    lo, hi = int(bits.min()), int(bits.max())
                    if lo == hi:
                        const[b] = lo
                per_layer.append(const)
            out[cls] = per_layer
        return out

    @torch.no_grad()
    def fingerprint(self):
        """Checksum of all target weights and stored arrays. Any single bit left flipped changes
        it, so comparing before and after a run catches a missed restore."""
        acc = []
        for l in self.layers:
            w = l.module.weight.data
            acc.append(int(w.view(torch.int16).to(torch.int32).sum()))
            for t in l.store.values():
                acc.append(zlib.crc32(t.contiguous().view(torch.uint8).numpy()))
        return acc

def fmt_value(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)
