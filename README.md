# LeJEPA-PCA

## Overview

LeJEPA-PCA is a self-supervised learning project that trains image models and visualizes output using PCA, supporting both RGB and infrared data.

---

## Architecture

- **Single Encoder + Projector (SigReg)**  
  The model uses a single encoder with a projection head, trained using a signal-regularization (SigReg) objective.

- **SigReg Objective**  
  Regularizes the latent space directly to encourage informative, non-degenerate representations.

- **Patch-Level Feature Learning**  
  Representations are learned at the patch level using a ResNet backbone, enabling spatially localized latent analysis.

## Installation

Minimal setup:

- **Python:** 3.10+
- **Dependencies:** PyTorch, NumPy, scikit-learn, joblib
- **CUDA:** Recommended for training but not required for analysis

```bash
pip install -r requirements.txt
```

## Training

### RGB Training
`python train.py --data_root ds/train/rgb_images`
### IR Training
`python train.py --use_ir --data_root ds/train/ir_images`

`--run_eval` flag can be added to trigger eval pipeline.

## Evaluation Pipeline
Evaluation focuses on latent behavior, not downstream task accuracy.

Metrics include:
- Latent prediction loss
- Cosine similarity between latent embeddings
- Squared distance in representation space

Evaluation can be triggered via: 
- `make rgb_eval`
- `make ir_eval`

## Representation Analysis & Visualization
For PCA visualization only, use 
`make rgb_viz NUM_VIZ=<int>`
`make ir_viz NUM_VIZ=<int>`

## End-to-End Workflow
Train → Evaluate → Dump Tokens → Fit PCA → Visualize
Optional one-command pipelines:
- `make rgb_all`- `make ir_all`

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
```
