# Multimodal Contrastive Learning

End-to-end project: synthetic image+caption pairs -> CLIP-style dual encoder -> retrieval and zero-shot evaluation. Everything is generated in-notebook (no downloads) and runs on CPU in well under 90 seconds.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`multimodal.ipynb`](multimodal.ipynb) | Image CNN + text transformer sharing one embedding space; Symmetric InfoNCE loss, learnable temperature, duplicate masking; Image-to-text, text-to-image retrieval and zero-shot classification; Compositional generalisation to unseen colour+shape combinations; Visualising the joint embedding space with PCA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/multimodal/multimodal.ipynb) |

## Module files (imported by the notebook)

| File | Contents |
|------|----------|
| `data.py` | coloured-shape images, templated captions, `Tokenizer` |
| `model.py` | `ImageEncoder`, `TextEncoder`, `CLIP` |
| `train.py` | `clip_loss`, `fit`, retrieval and zero-shot helpers |
| `test_multimodal.py` | unit tests |

## Run it

```bash
cd 06_end_to_end_projects/multimodal
python -m unittest -q        # module tests
jupyter lab multimodal.ipynb
```

On Colab the first cell clones the repo and changes into this folder so the module imports work.
