# Image Classification

End-to-end project: synthetic noisy shape images (circle/square/triangle/cross) -> small CNN -> train/eval -> checkpoint -> inference + serving stub. Everything is generated in-notebook (no downloads) and runs on CPU in well under 90 seconds.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`image_classification.ipynb`](image_classification.ipynb) | Generating and inspecting synthetic image data; Shape-tracing a CNN and the one-batch overfit sanity check; Train/eval loop, learning curves, confusion matrix, error analysis; Saving and safely reloading a checkpoint (`weights_only=True`); A `Predictor` class with request validation (optional FastAPI endpoint) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/image_classification/image_classification.ipynb) |

## Module files (imported by the notebook)

| File | Contents |
|------|----------|
| `data.py` | synthetic shape generator, split |
| `model.py` | `SmallCNN`, checkpoint save/load, `Predictor` |
| `train.py` | train/eval loops, confusion matrix |
| `test_image_classification.py` | unit tests |

## Run it

```bash
cd 06_end_to_end_projects/image_classification
python -m unittest -q        # module tests
jupyter lab image_classification.ipynb
```

On Colab the first cell clones the repo and changes into this folder so the module imports work.
