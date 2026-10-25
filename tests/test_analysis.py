import numpy as np
import pandas as pd

from scalebit.analysis import clopper_pearson, fit_table, protection_curve


def test_clopper_pearson_zero_of_200():
    lo, hi = clopper_pearson(0, 200)
    assert lo == 0 and abs(hi - 0.0183) < 1e-3


def _pos():
    rows = []
    for cls, width, fpw, meta, avf in [("element", 4, 1.0, False, 0.0),
                                        ("scale", 8, 1 / 32, True, 0.0)]:
        for b in range(width):
            a = 0.5 if (cls == "scale" and b == 7) else avf
            rows.append({"model": "m", "task": "t", "format": "mxfp4", "dtype": "float16",
                         "class": cls, "bit": b, "avf": a, "avf_lo": a * 0.8,
                         "avf_hi": a + 0.01, "fields_per_weight": fpw, "metadata": meta})
    return pd.DataFrame(rows)


def test_fit_and_curve():
    f = fit_table(_pos()).iloc[0]
    assert np.isclose(f.fit, 0.5 / 32)
    assert np.isclose(f.bits_per_weight, 4.25)
    assert np.isclose(f.metadata_fit_share, 1.0)
    c = protection_curve(_pos())
    last = c.iloc[-1]
    assert last.protected == "scale:b7" and np.isclose(last.protected_bits_per_weight, 1 / 32)
    assert np.isclose(last.fit_left, 0.0)
