# Recommendation System

End-to-end project: synthetic user-item interactions -> two-tower embedding model -> in-batch negatives -> recall@k. Everything is generated in-notebook (no downloads) and runs on CPU in well under 90 seconds.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`recommendation_system.ipynb`](recommendation_system.ipynb) | Implicit feedback, sparsity and per-user leave-k-out splits; Two-tower retrieval with L2-normalised embeddings and a temperature; In-batch softmax loss, false-negative masking, logQ popularity correction; recall@k against random and popularity baselines; Serving with a precomputed item index (optional FAISS) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/recommendation_system/recommendation_system.ipynb) |

## Module files (imported by the notebook)

| File | Contents |
|------|----------|
| `data.py` | interaction simulator, split, batching |
| `model.py` | `TwoTower`, `recall_at_k`, popularity baseline |
| `train.py` | in-batch softmax loss, `fit` |
| `test_recommendation_system.py` | unit tests |

## Run it

```bash
cd 06_end_to_end_projects/recommendation_system
python -m unittest -q        # module tests
jupyter lab recommendation_system.ipynb
```

On Colab the first cell clones the repo and changes into this folder so the module imports work.
