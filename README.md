# Deepfake-Detection-CNN-ViT

Two architectures, one goal: An EfficientNetB4 trained on 257K diverse images (AUC-ROC 0.9751) and a forensic Vision Transformer trained on 105K images with multi-layer attention maps, both deployed live on HuggingFace Spaces.

---

## Overview

This project develops and compares two deep learning approaches for binary classification of real versus AI-generated images. Built for MSBC 5190 Modern AI: Introduction to AI for Business at the University of Colorado Boulder (Leeds School of Business), Spring 2026.

The core question: **Can a model reliably distinguish authentic photographs from AI-generated fakes   and explain why?**

Both models are live. Both have documented failure modes. No results were cherry-picked.

---

## The Problem

Generative AI tools including Stable Diffusion, Midjourney, and GANs have made photorealistic synthetic images trivially easy to produce. Fabricated images of public figures, fake news photographs, and AI-generated evidence spread faster than manual fact-checking can respond. This project builds accessible, explainable detectors that tell users not just *what* was decided, but *why*.

---

## Models

### EfficientNetB4 V3   Primary CNN Model
- **Architecture:** EfficientNetB4 pre-trained on ImageNet, fine-tuned via two-phase transfer learning
- **Training Data:** ~257,283 images across 5 datasets (CIFAKE, 140k Real & Fake Faces, OpenForensics, Natural Images HD, HEMG/FakeShield)
- **Training Platform:** Lightning AI (Tesla T4 GPU, ~20 hours)
- **Phase 1:** 10 epochs, frozen backbone, Adam lr=1e-3
- **Phase 2:** 25 epochs, top layers unfrozen, Cosine Decay lr from 1e-5
- **Explainability:** Grad-CAM spatial saliency heatmaps
- **Known limitation:** Fails on Midjourney V5/V6 images   absent from training data

### Vision Transformer (ViT-XAI)   Forensic Face Specialist
- **Architecture:** google/vit-base-patch16-224-in21k   86M parameters, 12 attention heads
- **Training Data:** ~105,000 images   CIFAKE (generalist, Stage 1) + 5,000 StyleGAN facial images (specialist, Stage 2)
- **Training:** AdamW optimizer, sequential transfer learning, Binary Cross-Entropy loss
- **Explainability:** 4-layer forensic attention visualization (texture → geometry → artifact search → final decision)
- **Known limitation:** High false-positive rate on real photographs due to StyleGAN specialist fine-tuning

### Ensemble Meta-Learner (Experimental   Not Deployed)
- Combined ViT-V1, ViT-V2 (continual learning variant), and EfficientNetB4 via logistic regression meta-learner
- Achieved 92.25% accuracy on 2,000-image blind validation set.
- Not deployed, generalization on out-of-distribution images was insufficient for production

---

## Results

| Model | AUC-ROC | Accuracy | Training Images | Platform |
|---|---|---|---|---|
| EfficientNetB4 V1 (baseline) | 0.9924 | 96.00% | ~100K (CIFAKE only) | Google Colab |
| EfficientNetB4 V3 (primary) | **0.9751** | **92.20%** | ~257K (5 datasets) | Lightning AI |
| ViT-XAI | 0.91 | ~86% real-world | ~105K | Lightning AI |

> V1's higher numbers reflect overfitting to CIFAKE, not generalization. V3's lower numbers represent meaningful improvement on harder, more diverse examples.

---

## Explainability

**Grad-CAM (V3):** Produces a spatial heatmap overlaid on the input image, highlighting regions that drove the real/fake classification   unnatural textures, blending artifacts, or malformed edges.

**Multi-Layer Attention (ViT):** Visualizes forensic attention across 4 layers progressively   Layer 1 (initial texture), Layer 4 (facial geometry), Layer 8 (artifact search), Layer 12 (final decision).

---

## Live Demos

| Model | HuggingFace Space |
|---|---|
| EfficientNetB4 V3 | [deepfake-detector-v3](https://huggingface.co/spaces/sajaaghareeb/deepfake-detector-v3) |
| ViT-XAI | [face-ai-detector-91auc](https://huggingface.co/Vishwajitm01/face-ai-detector-91auc) |

Each app returns: binary verdict (Real / AI-Generated), confidence score, and visual explainability overlay.

---

## Repository Structure

```
├── EfficientNetB4_V3_Notebook.ipynb      # CNN training, evaluation, Grad-CAM
├── ViT_XAI_Notebook.ipynb                # Vision Transformer training and attention maps
├── Ensemble_Notebook.ipynb               # Meta-learner ensemble experiment
├── Deepfake_Detection_Report.pdf         # Full written report
├── Deepfake_Detection_Slides.pptx        # Presentation slides
├── accuracy_logs/                        # Training history and metric logs
```

> Frozen model weights are hosted on HuggingFace links above.

---

## Model Limitations

This project documents where models break, not just where they succeed:

- **V3 Midjourney blind spot**   Misclassified a Midjourney fox image as Real (52.8% confidence). Cause: Midjourney dataset deleted from HuggingFace before training. Predictable and reproducible.
- **V4 catastrophic forgetting**   Fine-tuning V3 on 3,000 Midjourney images destroyed broad generalization. Reverted to V3.
- **ViT ** .
- **ViT-V2 continual learning failure**   85-87% validation accuracy collapsed to 41.79% on independent test data.

---

## Team

Vishwajit Mohan Kumar · Retaj Muhsen · Saja Ali · Trew Mundy · Ima Mervin
MS Business Analytics   University of Colorado Boulder, Leeds School of Business

---

## Tools & Skills

`Python` `TensorFlow` `PyTorch` `EfficientNetB4` `Vision Transformer` `Transfer Learning` `Grad-CAM` `Explainable AI` `Gradio` `HuggingFace Spaces` `Deep Learning` `Computer Vision`
