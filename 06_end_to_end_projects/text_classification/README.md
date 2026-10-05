# Text Classification

End-to-end project: synthetic sentiment text with negation -> tokenizer -> transformer encoder classifier written from scratch. Everything is generated in-notebook (no downloads) and runs on CPU in well under 90 seconds.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`text_classification.ipynb`](text_classification.ipynb) | Word-level tokenizer, padding and pad masks; Multi-head self-attention written by hand, with sanity checks; Why a bag-of-words baseline fails on negation and attention does not; Slicing evaluation by subgroup, inspecting `<cls>` attention; Robustness: unknown words, empty input | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/text_classification/text_classification.ipynb) |

## Module files (imported by the notebook)

| File | Contents |
|------|----------|
| `data.py` | corpus generator and `Tokenizer` |
| `model.py` | `SelfAttention`, pre-LN `Block`, `TransformerClassifier` |
| `train.py` | batching, `fit`, `evaluate`, `predict` |
| `test_text_classification.py` | unit tests |

## Run it

```bash
cd 06_end_to_end_projects/text_classification
python -m unittest -q        # module tests
jupyter lab text_classification.ipynb
```

On Colab the first cell clones the repo and changes into this folder so the module imports work.
