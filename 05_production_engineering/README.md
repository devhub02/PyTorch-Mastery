# 🔥 05 — Production Engineering

Taking a model from a notebook experiment to something you can train at scale, reproduce,
test, export and serve. Every notebook here covers one concept with explanation, runnable
code (CPU-only, synthetic data, no downloads), common bugs, and exercises.

> **Click a notebook name** to view on GitHub, or the **Colab badge** to open directly in Google Colab — no setup required.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`mixed_precision.ipynb`](mixed_precision.ipynb) | `torch.autocast` (bf16 on CPU), `GradScaler` logic, fp16 underflow, dtype ranges, speed/memory trade-offs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/mixed_precision.ipynb) |
| [`distributed_training.ipynb`](distributed_training.ipynb) | DataParallel vs DDP vs FSDP, a real 2-process gloo DDP run on CPU with verified gradient sync, `DistributedSampler` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/distributed_training.ipynb) |
| [`experiment_tracking.ipynb`](experiment_tracking.ipynb) | JSON-lines logger, full-state checkpoints (model/optimizer/scheduler/RNG), exact resume test, best-k checkpoints, wandb/mlflow (guarded) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/experiment_tracking.ipynb) |
| [`model_serving.ipynb`](model_serving.ipynb) | Dynamic request batching with a queue, latency percentiles vs throughput, validation, FastAPI/TorchServe templates | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/model_serving.ipynb) |
| [`onnx_torchscript.ipynb`](onnx_torchscript.ipynb) | `torch.jit.trace`/`script`, `torch.export`, ONNX export (guarded), numerical output comparison and export pitfalls | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/onnx_torchscript.ipynb) |
| [`testing_ml_code.ipynb`](testing_ml_code.ipynb) | Shape, overfit-one-batch, determinism, gradient-flow and invariance tests run with `unittest`; mutation checks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/testing_ml_code.ipynb) |
| [`profiling_optimization.ipynb`](profiling_optimization.ipynb) | `torch.profiler`, memory tracking, common performance bugs, `torch.compile` with eager fallback | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/profiling_optimization.ipynb) |
| [`reproducibility_config.ipynb`](reproducibility_config.ipynb) | Seeding, deterministic algorithms, DataLoader seeding, dataclass/YAML configs, run manifests, Hydra (guarded) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/05_production_engineering/reproducibility_config.ipynb) |

> Optional libraries (`onnx`, `onnxruntime`, `wandb`, `mlflow`, `fastapi`, `hydra-core`, `pytest`) are used inside `try/except ImportError` blocks: install them to see extra output, but every notebook runs top to bottom without them.

---

**What's inside each notebook:**

| Section | What You Get |
|---------|--------------|
| 📖 Concept Overview | Plain English explanation of what it is and why it exists |
| 🔢 Runnable Code | Numbered sections — minimal, heavily commented, actually executed |
| ⚠️ Common Bugs | The mistakes everyone makes here + how to diagnose them |
| ✏️ Exercises | Practice problems with a scratch code cell and a worked solution |
