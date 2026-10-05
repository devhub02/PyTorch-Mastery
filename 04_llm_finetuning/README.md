# 🔥 04 — LLM Finetuning

From tokenizers and datasets to full finetuning, parameter-efficient methods (LoRA / QLoRA), instruction tuning, preference optimization, quantization, fast inference and evaluation. Everything runs on CPU with tiny models and synthetic data generated inside the notebook, and no downloads are needed. Optional libraries (`transformers`, `datasets`, `tokenizers`, `accelerate`) are used only inside `try/except ImportError` blocks.

> **Click a notebook name** to view on GitHub, or the **Colab badge** to open directly in Google Colab — no setup required.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`huggingface_ecosystem.ipynb`](huggingface_ecosystem.ipynb) | Tokenizers (char + BPE from scratch), datasets-style `map`/`filter`, collation with masks and `-100`, config + `save_pretrained`, generation, a mini `Accelerator` (guarded real HF calls) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/huggingface_ecosystem.ipynb) |
| [`full_finetuning.ipynb`](full_finetuning.ipynb) | Full-parameter finetuning of a tiny GPT, forgetting, LR sensitivity, memory math (params / grads / Adam states), gradient checkpointing, checkpoint resume | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/full_finetuning.ipynb) |
| [`lora_qlora.ipynb`](lora_qlora.ipynb) | LoRA layer from scratch (rank, alpha, merge/unmerge, adapter swapping), SVD of the finetuning delta, NF4-like blockwise 4-bit quantization, QLoRA layer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/lora_qlora.ipynb) |
| [`prompt_instruction_tuning.ipynb`](prompt_instruction_tuning.ipynb) | Alpaca and ChatML formats, hand-written chat templates, response-only loss masking, padding vs packing (block-diagonal mask), template mismatch, soft-prompt tuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/prompt_instruction_tuning.ipynb) |
| [`rlhf_dpo.ipynb`](rlhf_dpo.ipynb) | Bradley-Terry reward models, PPO clipped objective with KL penalty, DPO loss and implicit reward | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/rlhf_dpo.ipynb) |
| [`quantization.ipynb`](quantization.ipynb) | int8 / int4, per-channel and block-wise scales, outliers, packing, quantizing a tiny LM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/quantization.ipynb) |
| [`inference_optimization.ipynb`](inference_optimization.ipynb) | KV cache, batching and padding, speculative decoding, speedup modelling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/inference_optimization.ipynb) |
| [`evaluation.ipynb`](evaluation.ipynb) | Perplexity, log-likelihood multiple choice, exact match, mini harness, bootstrap confidence intervals, contamination checks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/04_llm_finetuning/evaluation.ipynb) |

> 📝 **Suggested order:** the first four notebooks build the finetuning toolkit (data, full finetuning, LoRA, instruction formats); the last four cover alignment, compression, serving and measurement.

---

**What's inside each notebook:**

| Section | What You Get |
|---------|--------------|
| 📖 Concept Overview | Plain English explanation of what it is and why it exists |
| 🔢 Runnable Code | Numbered sections — minimal, heavily commented, actually executed |
| ⚠️ Common Bugs | The mistakes everyone makes here + how to diagnose them |
| ✏️ Exercises | Practice problems with a scratch code cell and a worked solution |
