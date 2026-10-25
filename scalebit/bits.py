"""Low-level helpers: XOR a bit inside a stored word, and the FP4/FP8 code tables."""

import torch

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def xor_bit(t, idx, bit):
    """Flip `bit` of the element t[idx] in place, whatever the dtype. Returns (old, new) raw words."""
    size = t.element_size()
    v = t.view(_INT_VIEW[size])
    nbits = 8 * size
    old = int(v[idx]) & ((1 << nbits) - 1)
    new = old ^ (1 << bit)
    signed = new - (1 << nbits) if (size > 1 and new >= 1 << (nbits - 1)) else new
    v[idx] = signed
    return old, new


# E2M1: code = sign<<3 | exp<<1 | mantissa. Magnitude index = code & 7.
E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def e2m1_table(device="cpu"):
    m = E2M1_MAG.to(device)
    return torch.cat([m, -m])


def e2m1_encode(x):
    """Round to nearest E2M1 value, ties to even mantissa, saturating at +-6."""
    mid = _E2M1_MID.to(x.device)
    a = x.abs().clamp(max=6.0)
    lo = torch.bucketize(a, mid, right=False)
    hi = torch.bucketize(a, mid, right=True)
    idx = torch.where((lo != hi) & (hi % 2 == 0), hi, lo)
    return (torch.signbit(x).to(torch.uint8) << 3) | idx.to(torch.uint8)


def e2m1_role(bit):
    return {0: "mantissa", 1: "exponent", 2: "exponent", 3: "sign"}[bit]


def pack_nibbles(codes):
    """[rows, n] uint8 codes (0..15) -> [rows, n/2]; even index in the low nibble."""
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def unpack_nibbles(packed):
    rows, half = packed.shape
    c = torch.empty(rows, half * 2, dtype=torch.uint8, device=packed.device)
    c[:, 0::2] = packed & 0x0F
    c[:, 1::2] = packed >> 4
    return c


# E4M3 (OCP FP8, "fn" variant): bias 7, max 448, no Inf, NaN = S.1111.111 (0x7F / 0xFF).
E4M3_MAX = 448.0


def e4m3_decode(raw):
    """uint8 raw bits -> float32 values (NaN for 0x7F/0xFF)."""
    return raw.view(torch.float8_e4m3fn).to(torch.float32)


def e4m3_encode(x):
    """float -> uint8 raw bits, round to nearest even, clamped to [-448, 448]."""
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_role(bit):
    return "sign" if bit == 7 else ("exponent" if bit >= 3 else "mantissa")


def float_role(bit, mant_bits, total_bits):
    if bit == total_bits - 1:
        return "sign"
    return "exponent" if bit >= mant_bits else "mantissa"
