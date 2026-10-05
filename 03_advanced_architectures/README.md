# 🔥 03 — Advanced Architectures

The architectures behind modern deep learning, each built from scratch in pure PyTorch on
small synthetic data (no downloads, CPU-friendly). Every notebook has an explanation,
runnable code, common bugs, and exercises.

> **Click a notebook name** to view on GitHub, or the **Colab badge** to open directly in Google Colab — no setup required.

| Notebook | What You'll Learn | Open |
|----------|-------------------|------|
| [`attention_transformers.ipynb`](attention_transformers.ipynb) | Scaled dot-product & multi-head attention, causal/padding masks, encoder & decoder blocks, a tiny char-level GPT trained from scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/03_advanced_architectures/attention_transformers.ipynb) |
| [`vision_transformers.ipynb`](vision_transformers.ipynb) | Patch embedding (= strided conv), [CLS] token, position embeddings, ViT trained on synthetic images, patch-size cost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/03_advanced_architectures/vision_transformers.ipynb) |
| [`resnet_efficientnet.ipynb`](resnet_efficientnet.ipynb) | Residual blocks, why skips help gradient flow, SE, depthwise-separable convs, MBConv, compound scaling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/03_advanced_architectures/resnet_efficientnet.ipynb) |
| [`gans.ipynb`](gans.ipynb) | Generator/discriminator dynamics on 1D and 2D toys, non-saturating loss, reading GAN losses, mode collapse, WGAN-GP penalty | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/03_advanced_architectures/gans.ipynb) |
| [`vaes.ipynb`](vaes.ipynb) | Autoencoders to VAEs: encoder/decoder, reparameterisation trick, ELBO, latent space sampling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/03_advanced_architectures/vaes.ipynb) |
| [`gnn.ipynb`](gnn.ipynb) | Graph neural networks: message passing, adjacency/normalisation, GCN-style layers on toy graphs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/03_advanced_architectures/gnn.ipynb) |
| [`diffusion_models.ipynb`](diffusion_models.ipynb) | Forward noising, denoising objective, noise schedules, sampling with a tiny denoiser | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/himanshu231204/PyTorch-Mastery/blob/main/03_advanced_architectures/diffusion_models.ipynb) |

> 💡 **Suggested order:** `attention_transformers` → `vision_transformers` → `resnet_efficientnet` → `gans` → `vaes` → `diffusion_models`, with `gnn` whenever you want to move beyond grids and sequences.

---

**What's inside each notebook:**

| Section | What You Get |
|---------|--------------|
| 📖 Concept Overview | Plain English explanation of what it is and why it exists |
| 🔢 Runnable Code | Numbered sections — minimal, heavily commented, actually executed |
| ⚠️ Common Bugs | The mistakes everyone makes here + how to diagnose them |
| ✏️ Exercises | Practice problems with a scratch code cell and a worked solution |
