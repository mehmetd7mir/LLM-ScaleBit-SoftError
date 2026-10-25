from .fp import FP
from .int4 import INT4
from .mxfp4 import MXFP4
from .nvfp4 import NVFP4

FORMATS = ("fp", "int4_rtn", "int4_gptq", "mxfp4", "nvfp4")


def make_format(name, dtype, gptq_id=None):
    if name == "fp":
        return FP(dtype)
    if name == "int4_rtn":
        return INT4(dtype, source="rtn")
    if name == "int4_gptq":
        if not gptq_id:
            raise ValueError("int4_gptq needs gptq_id")
        return INT4(dtype, source="gptq", gptq_id=gptq_id)
    if name == "mxfp4":
        return MXFP4(dtype)
    if name == "nvfp4":
        return NVFP4(dtype)
    raise ValueError(f"unknown format {name!r}; choose from {FORMATS}")
