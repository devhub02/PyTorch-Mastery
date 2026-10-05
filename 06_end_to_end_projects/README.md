# 06 — End-to-End Projects

Five small but complete projects, each with a notebook, a tested Python module the notebook imports, and a README. All data is synthetic and generated in-notebook, models are written from scratch in pure PyTorch, and everything runs on CPU.

> **Click a project name** to browse it, or the **Colab badge** to open the notebook directly.

| Project | Task and Data | Model and Key Ideas | Metric | Open |
|---------|---------------|---------------------|--------|------|
| [Image classification](image_classification/) | Synthetic noisy shape images (4 classes) | Small CNN, global average pooling, checkpoint + `Predictor` serving stub | Accuracy, confusion matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/image_classification/image_classification.ipynb) |
| [Text classification](text_classification/) | Synthetic sentiment sentences with negation | Word tokenizer, from-scratch transformer encoder, pad masks, `<cls>` pooling | Accuracy vs bag-of-words baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/text_classification/text_classification.ipynb) |
| [LLM pretraining + LoRA finetuning](llm_finetuning/) | Toy sentence corpus, then instruction pairs | Character-level tiny GPT, LoRA adapters (freeze, zero-init, merge), answer-only loss | Exact-match, forgetting check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/llm_finetuning/llm_finetuning.ipynb) |
| [Recommendation system](recommendation_system/) | Synthetic user-item clicks from latent taste groups | Two-tower embeddings, in-batch negatives, logQ correction | recall@k vs popularity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/recommendation_system/recommendation_system.ipynb) |
| [Multimodal contrastive learning](multimodal/) | Synthetic image + caption pairs (coloured shapes) | CLIP-style dual encoder, symmetric InfoNCE, learnable temperature | Retrieval R@k, zero-shot accuracy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/06_end_to_end_projects/multimodal/multimodal.ipynb) |

---

**Every project folder contains:**

| File | Purpose |
|------|---------|
| `<project>.ipynb` | Main notebook: concept notes, 11-13 numbered sections, common bugs, exercises with solutions |
| `data.py` / `model.py` / `train.py` | Small module the notebook imports |
| `test_<project>.py` | `unittest` tests (`python -m unittest -q` inside the folder) |
| `README.md` | Summary and Colab link |
