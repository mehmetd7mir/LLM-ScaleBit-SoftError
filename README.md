# LLM-ScaleBit-SoftError

Single-bit fault injection into the stored weights of 4-bit LLMs, **including the metadata**
(scales, zero points) that most fault-injection studies leave out. The goal is a fair,
bit-count-aware comparison of 4-bit formats against an FP16 baseline.

## Formats

| name | elements | metadata | bits/weight |
|---|---|---|---|
| `fp` | FP16 or BF16 weights | none | 16 |
| `int4_rtn` | INT4, group 128 | FP16 scale + 4-bit zero point per group | 4.156 |
| `int4_gptq` | same layout, loaded from a published GPTQ checkpoint | same | 4.156 |
| `mxfp4` | E2M1, block 32 (OCP MX v1.0) | E8M0 scale per block | 4.25 |
| `nvfp4` | E2M1, block 16 | E4M3 scale per block + FP32 scale per tensor | 4.5 |

`int4_rtn`, `mxfp4` and `nvfp4` use round-to-nearest, so the main comparison differs only in
the format. `int4_gptq` shares the `int4_rtn` layout and isolates the effect of a
calibrated quantizer. Checks against reference implementations:

- MXFP4 matches torchao bit for bit.
- NVFP4 matches compressed-tensors (llm-compressor) in scales and values.
- INT4 reads the AutoGPTQ v1 layout, with the stored zero point plus one.

## Fault model

- **One trial:** one bit of one stored value flips, one question is evaluated, and the bit is
  restored. After every trial the model is bit-exact again.
- **Targets:** all linear layers inside the transformer blocks (q, k, v, o, gate, up, down).
- **Sampling:** each bit position of each class (for example the 8 bits of an MXFP4 scale) gets
  the same number of trials. Within a trial the layer is picked in proportion to its number of
  fields, then the field and the question are drawn uniformly.
- **Outcomes:**
  - `masked`: same answer as the fault-free run.
  - `sdc`: the answer changed silently.
  - `visible`: NaN or Inf appeared.
  - `benign`: GSM8K only; the text changed but the final answer did not.

The stored arrays live on the CPU and the dequantized weights live on the GPU. A flip
recomputes only the weights that depend on the flipped bit: one element, one block or group,
or, for the NVFP4 tensor scale, the whole layer.

## Tasks

- **`mmlu`:** tinyMMLU, 100 questions. One forward pass picks the answer from the logits of
  A/B/C/D.
- **`gsm8k`:** tinyGSM8K, 100 questions, greedy decoding. A faulty run is first checked with a
  single teacher-forced pass over the golden tokens. If every argmax matches, the output is
  identical and no generation is needed. Otherwise generation continues from the first
  divergent token and the final number is compared with the golden answer.

## Running on Colab

Open `notebooks/run_experiment.ipynb` in Colab with an L4 runtime, set the config cell and run
all cells. The notebook clones this repository, writes results to Google Drive and resumes
after a disconnect. `notebooks/analyze.ipynb` builds the tables and plots from all runs.

Results layout:

```
<results_root>/<study>/<model>/<task>/<format>_<dtype>/
    run_info.json       settings, versions, GPU, git commit
    golden.jsonl        fault-free output per question
    trials_<class>.csv  one row per trial
```

## Local tests

```
pip install -r requirements.txt pytest
python -m pytest tests
```

The tests run on the CPU in a few seconds. `float32` is accepted as a compute dtype for quick
CPU runs only; the experiments use `float16` (and `bfloat16` for the dtype sub-study).

## Status

Research code for a paper in preparation (Ankara Yıldırım Beyazıt University).
