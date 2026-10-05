# LLM Pretraining and LoRA Finetuning

End-to-end project: tiny character-level GPT pretrained on a toy corpus, then LoRA-style instruction finetuning, then generation. Everything is generated in-notebook (no downloads) and runs on CPU in well under 90 seconds.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`llm_finetuning.ipynb`](llm_finetuning.ipynb) | Causal self-attention GPT from scratch, initial-loss and causality checks; Pretraining, greedy vs temperature sampling; Instruction pairs and answer-only loss masking; LoRA: low-rank adapters, zero init, freezing, merging, adapter-only storage; Measuring exact-match and catastrophic forgetting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/llm_finetuning/llm_finetuning.ipynb) |

## Module files (imported by the notebook)

| File | Contents |
|------|----------|
| `data.py` | toy corpus, `CharTokenizer`, LM and SFT batching |
| `model.py` | `TinyGPT`, `LoRALinear`, `apply_lora`, `merge_lora` |
| `train.py` | `pretrain`, `finetune`, `complete`, `exact_match` |
| `test_llm_finetuning.py` | unit tests |

## Run it

```bash
cd 06_end_to_end_projects/llm_finetuning
python -m unittest -q        # module tests
jupyter lab llm_finetuning.ipynb
```

On Colab the first cell clones the repo and changes into this folder so the module imports work.
