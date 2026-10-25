import numpy as np
import pandas as pd

from scalebit.analysis import _protect, clopper_pearson, fit_table, protection_curve


def test_clopper_pearson_zero_of_200():
    lo, hi = clopper_pearson(0, 200)
    assert lo == 0 and abs(hi - 0.0183) < 1e-3


def _pos(n=300):
    rows = []
    for cls, width, fpw, meta in [("element", 4, 1.0, False), ("scale", 8, 1 / 32, True)]:
        for b in range(width):
            k = n if (cls == "scale" and b == 7) else 0
            lo, hi = clopper_pearson(k, n)
            rows.append({"model": "m", "task": "t", "format": "mxfp4", "dtype": "float16",
                         "class": cls, "bit": b, "n": n, "fail": k, "sdc": 0, "visible": k,
                         "benign": 0, "avf": k / n, "avf_lo": lo, "avf_hi": hi,
                         "fields_per_weight": fpw, "metadata": meta})
    return pd.DataFrame(rows)


def test_fit_uses_pooled_interval():
    f = fit_table(_pos()).iloc[0]
    assert np.isclose(f.fit, 1 / 32)
    assert np.isclose(f.bits_per_weight, 4.25)
    assert np.isclose(f.metadata_fit_share, 1.0)
    # element upper bound from 1200 pooled trials, not 4 x the 300-trial bound
    assert np.isclose(f.fit_hi - (8 / 32) * clopper_pearson(300, 2400)[1],
                      4 * clopper_pearson(0, 1200)[1])


def test_curve():
    c = protection_curve(_pos())
    last = c.iloc[-1]
    assert last.protected == "scale:b7" and np.isclose(last.protected_bits_per_weight, 1 / 32)
    assert np.isclose(last.fit_left, 0.0)


def test_protect_marks_constant_bits_masked():
    df = pd.DataFrame({"bit_class": ["scale", "scale", "element"], "layer": [0, 1, 0],
                       "bit": [7, 7, 3], "outcome": ["visible", "visible", "sdc"]})
    const = {"scale": [{"7": 0}, {}]}
    out = _protect(df, const)
    assert list(out.outcome) == ["masked", "visible", "sdc"]
