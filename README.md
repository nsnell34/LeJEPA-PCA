# LeJEPA-PCA

**PyTorch implementation of LeJEPA with PCA-based representation analysis and visualization.**

This repository focuses on learning **patch-level latent representations** using a JEPA-style self-supervised objective, with first-class support for **representation inspection and visualization**. It supports both **RGB and Infrared (IR)** modalities and emphasizes understanding *what the model learns*, not just how fast the loss drops. Visualization is treated as a core component rather than a post-hoc add-on.

---

## Overview

LeJEPA-PCA implements a Latent JEPA / I-JEPA–style self-supervised model for learning patch-level representations from images. The goal is to train modality-agnostic encoders while enabling deep inspection of learned latent structure through PCA-based visualization. Both RGB and IR pipelines are supported, with modality-specific representations. The project prioritizes representation analysis over downstream task accuracy.

---

## Architecture

- **Single Encoder + Projector (SigReg)**  
  The model uses a single encoder with a projection head, trained using a signal-regularization (SigReg) objective rather than a dual encoder or EMA teacher.

- **SigReg Objective**  
  Regularizes the latent space directly to encourage informative, non-degenerate representations without requiring a target network or stop-gradient tricks.

- **Patch-Level Feature Learning**  
  Representations are learned at the patch level using a ResNet backbone, enabling spatially localized latent analysis.

## Installation

Minimal setup:

- **Python:** 3.10+
- **Dependencies:** PyTorch, NumPy, scikit-learn, joblib
- **CUDA:** Recommended for training but not required for analysis

```bash
pip install -r requirements.txt
Training
Training is fully self-supervised — no labels are required.
RGB Training
python train.py --data_root ds/train/rgb_images
Infrared (IR) Training
python train.py --use_ir --data_root ds/train/ir_images
Notes
Uses a single encoder + projector trained with SigReg
Supports RGB and IR via a shared architecture
Model checkpoints are saved to ckpts/
Evaluation Pipeline
Evaluation focuses on latent behavior, not downstream task accuracy.
Metrics include:
Latent prediction loss
Cosine similarity between latent embeddings
Squared distance in representation space
Note: No classification accuracy or supervised metrics are used.
make rgb_eval
make ir_eval
Representation Analysis & Visualization
This is a core focus of the repository.
What Happens
Patch-level tokens are extracted from the trained encoder
PCA is fit once on the saved patch embeddings
The same PCA is reused for all visualizations
Visualizations reflect latent semantics, not pixel colors
make rgb_viz NUM_VIZ=10
make ir_viz NUM_VIZ=10
Interpretation Notes
Blocky outputs arise from patch-level representations rather than pixel-level predictions
Modality-specific PCA is used because RGB and IR occupy different latent distributions
Colors encode directions in latent space, not image intensity or RGB values
End-to-End Workflow
Train → Evaluate → Dump Tokens → Fit PCA → Visualize
Optional one-command pipelines:
make rgb_all
make ir_all

## Citations

```
@misc{balestriero2025lejepaprovablescalableselfsupervised,
      title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}, 
      author={Randall Balestriero and Yann LeCun},
      year={2025},
      eprint={2511.08544},
      archivePrefix={arXiv},```
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.08544}, 
}
