import pytest
import torch

from scalebit.bits import e2m1_encode, e2m1_table, e4m3_decode, xor_bit
from scalebit.formats import int4, make_format, mxfp4, nvfp4

DTYPES = [torch.float16, torch.bfloat16]


def linear(out_f=64, in_f=256, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(in_f, out_f, bias=False)
    lin.weight.data = (torch.randn(out_f, in_f) * 0.02).to(dtype)
    return lin


def same_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_xor_bit_all_widths():
    for dt, bits in [(torch.uint8, 8), (torch.int16, 16), (torch.int32, 32), (torch.float16, 16),
                     (torch.float32, 32)]:
        t = torch.zeros(3, dtype=dt)
        for b in range(bits):
            xor_bit(t, (1,), b)
            xor_bit(t, (1,), b)
        assert torch.equal(t, torch.zeros(3, dtype=dt))
    t = torch.zeros(1, dtype=torch.int32)
    xor_bit(t, (0,), 31)
    assert int(t[0]) == -2 ** 31


def test_e2m1_rounding():
    tab = e2m1_table()
    assert torch.equal(tab[e2m1_encode(tab).long()], tab)
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.3, 100.0])
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -0.5, 6.0])
    assert torch.equal(tab[e2m1_encode(x).long()], want)


def test_mxfp4_scale_rule_and_nan():
    w = torch.randn(4, 64) * 0.02
    codes, scales = mxfp4.quantize(w)
    for o in range(4):
        for b in range(2):
            s = 2.0 ** (int(scales[o, b]) - 127)
            assert 4 * s <= w[o, b * 32:(b + 1) * 32].abs().max() < 8 * s
    sc = scales.clone()
    sc[0, 0] = 255
    d = mxfp4.dequantize(codes, sc, torch.float32)
    assert torch.isnan(d[0, :32]).all() and not torch.isnan(d[0, 32:]).any()


def test_nvfp4_follows_report():
    w = torch.randn(8, 64) * 0.02
    codes, scales, s_dec = nvfp4.quantize(w)
    amax = w.abs().max()
    assert torch.isclose(s_dec[0], amax / (6 * 448))
    # the block holding the tensor amax gets scale 448 = 0x7E; one LSB flip makes it NaN
    o, i = divmod(int(w.abs().argmax()), 64)
    assert int(scales[o, i // 16]) == 0x7E
    sc = scales.clone()
    xor_bit(sc, (o, i // 16), 0)
    d = nvfp4.dequantize(codes, sc, s_dec, torch.float32)
    blk = slice(i // 16 * 16, i // 16 * 16 + 16)
    assert torch.isnan(d[o, blk]).all() and torch.isfinite(d[o, :blk.start]).all()
    # every block's amax lands near 6 after scaling
    deq = nvfp4.dequantize(codes, scales, s_dec, torch.float32)
    assert float((deq - w).norm() / w.norm()) < 0.15
    assert torch.equal(e4m3_decode(torch.tensor([0x7E], dtype=torch.uint8)), torch.tensor([448.0]))


def test_int4_pack_and_formula():
    in_f, out_f, gs = 256, 16, 128
    q = torch.randint(0, 16, (in_f, out_f))
    z = torch.full((in_f // gs, out_f), 7)
    sc = (torch.rand(in_f // gs, out_f) * 0.01 + 0.001).half()
    qw, qz = int4._pack_rows(q), int4._pack_cols(z)
    assert torch.equal(int4.unpack_qweight(qw), q.int())
    assert torch.equal(int4.unpack_qzeros(qz), z.int())
    g_idx = torch.arange(in_f) // gs
    w = int4.dequantize(qw, sc, qz, g_idx, 1, torch.float32)
    assert abs(float(w[3, 200]) - float(sc[1, 3].float() * (q[200, 3] - 8))) < 1e-6


def test_int4_rtn_matches_autogptq_rule():
    w = torch.randn(16, 256) * 0.02
    qw, sc, qz, g_idx = int4.quantize_rtn(w)
    xmax = w.t().reshape(2, 128, 16).abs().amax(1)
    assert torch.allclose(sc.float(), (xmax / 7.5).half().float())
    assert (int4.unpack_qzeros(qz) == 7).all()
    deq = int4.dequantize(qw, sc, qz, g_idx, 1, torch.float32)
    assert float((deq - w).norm() / w.norm()) < 0.15


@pytest.mark.parametrize("name", ["fp", "int4_rtn", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_flip_restore_exact(name, dtype):
    lin = linear(dtype=dtype)
    f = make_format(name, dtype)
    f.prepare(None, [("m.q_proj", lin)])
    before, fp0 = lin.weight.data.clone(), f.fingerprint()
    for cls, width in f.classes.items():
        n = f.field_counts(cls)[0]
        for field in sorted({0, 3, n // 2, n - 1}):
            for b in range(width):
                h = f.flip(0, cls, field, b)
                f.restore(h)
                assert same_bits(lin.weight.data, before), (cls, field, b)
    assert f.fingerprint() == fp0


@pytest.mark.parametrize("name,cls,size", [("mxfp4", "scale", 32), ("nvfp4", "scale", 16),
                                           ("int4_rtn", "scale", 128), ("int4_rtn", "zero", 128)])
def test_metadata_flip_touches_one_block(name, cls, size):
    lin = linear()
    f = make_format(name, torch.float16)
    f.prepare(None, [("m.q_proj", lin)])
    before = lin.weight.data.clone()
    bit = f.classes[cls] - 2 if cls == "scale" else 3
    h = f.flip(0, cls, 5, bit)
    changed = (lin.weight.data != before) & ~torch.isnan(before)
    rows = changed.any(1).nonzero().flatten().tolist()
    cols = changed.any(0).nonzero().flatten()
    f.restore(h)
    assert len(rows) == 1
    assert cols.numel() <= size and int(cols.max() - cols.min()) < size


@pytest.mark.parametrize("name", ["int4_rtn", "mxfp4", "nvfp4"])
def test_element_flip_touches_one_weight(name):
    lin = linear()
    f = make_format(name, torch.float16)
    f.prepare(None, [("m.q_proj", lin)])
    before = lin.weight.data.clone()
    field = 7 * 256 + 100
    h = f.flip(0, "element", field, 3)
    diff = (lin.weight.data != before).nonzero().tolist()
    f.restore(h)
    assert diff == [[7, 100]]


def test_nvfp4_tensor_scale_touches_whole_layer():
    lin = linear()
    f = make_format("nvfp4", torch.float16)
    f.prepare(None, [("m.q_proj", lin)])
    before = lin.weight.data.clone()
    h = f.flip(0, "tensor_scale", 0, 30)
    frac = float(((lin.weight.data != before) & (before != 0)).float().sum() / (before != 0).sum())
    f.restore(h)
    assert frac > 0.99 and same_bits(lin.weight.data, before)


def test_int4_unsorted_g_idx():
    """desc_act checkpoints store a permuted g_idx; a scale flip must hit exactly its rows."""
    torch.manual_seed(1)
    in_f, out_f, gs = 256, 16, 128
    lin = torch.nn.Linear(in_f, out_f, bias=False)
    lin.weight.data = (torch.randn(out_f, in_f) * 0.02).half()
    f = int4.INT4(torch.float16, "rtn")
    f.prepare(None, [("m.q_proj", lin)])
    layer = f.layers[0]
    perm = torch.randperm(in_f)
    layer.store["g_idx"] = (torch.arange(in_f) // gs)[perm]
    layer.group_rows = [torch.nonzero(layer.store["g_idx"] == g).flatten() for g in range(2)]
    s = layer.store
    lin.weight.data = int4.dequantize(s["qweight"], s["scales"], s["qzeros"], s["g_idx"], 1,
                                      torch.float16)
    before = lin.weight.data.clone()
    h = f.flip(0, "scale", 1 * out_f + 4, 14)
    cols = set((lin.weight.data[4] != before[4]).nonzero().flatten().tolist())
    f.restore(h)
    assert cols <= set(layer.group_rows[1].tolist()) and len(cols) > 100
    assert same_bits(lin.weight.data, before)


def test_nvfp4_matches_llm_compressor():
    """Independent reference: compressed-tensors (llm-compressor) NVFP4 RTN."""
    ct = pytest.importorskip("compressed_tensors.quantization")
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam
    from scalebit.bits import e4m3_decode
    args = ct.preset_name_to_scheme("NVFP4", ["Linear"]).weights
    torch.manual_seed(0)
    w = torch.randn(64, 512) * 0.03 * (1 + 5 * torch.rand(64, 1))
    gs = generate_gparam(w.min(), w.max())
    v = w.reshape(64, 32, 16)
    scale, zp = calculate_qparams(v.amin(-1), v.amax(-1), args, global_scale=gs)
    ref = fake_quantize(w, scale, zp, args, global_scale=gs)
    codes, sc, s_dec = nvfp4.quantize(w)
    assert torch.equal(scale.float(), e4m3_decode(sc))
    assert torch.equal(nvfp4.dequantize(codes, sc, s_dec, torch.float16), ref.half())


def test_constant_bits_found_and_single_field_skipped():
    lin = linear()
    f = make_format("mxfp4", torch.float16)
    f.prepare(None, [("m.q_proj", lin)])
    const = f.constant_bits()["scale"][0]
    assert const.get(7) == 0          # small weights never set the E8M0 MSB
    g = make_format("nvfp4", torch.float16)
    g.prepare(None, [("m.q_proj", linear())])
    assert g.constant_bits(["tensor_scale"])["tensor_scale"][0] == {}
