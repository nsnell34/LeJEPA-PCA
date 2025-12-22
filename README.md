# LeJEPA-PCA
Pytorch LeJEPA implementation utilizing PCA for data visualization 

## Project Overview

- What LeJEPA is (Latent JEPA / I-JEPA–style self-supervised model)
- High-level goal of the repo
- Learning patch-level representations
- Supporting RGB and IR modalities
- Emphasis on representation analysis & visualization (not just loss curves)
- Keep this to ~3–5 sentences.

## Key Ideas / Architecture

Brief description of:

- Context encoder + target encoder (EMA)
- Predictor head
- Patch-level feature extraction (ResNet backbone)
- Why JEPA (no negatives, no reconstruction)
- How collapse is avoided (EMA + predictor asymmetry)
- No equations — conceptual only.

## Repository Structure

Explain where to look:

lejepa/
├── train.py            # training loop
├── eval/               # evaluation + metrics
├── pca/                # token dumping, PCA fitting, visualization
├── transforms/         # data augmentations
├── tokens/             # saved patch embeddings (.npy)
├── joblibs/            # fitted PCA objects (.joblib)
├── viz/                # visualization outputs
├── ckpts/              # model checkpoints

## Installation

Minimal setup steps:

Python version
Install dependencies
(Optional) CUDA note

`pip install -r requirements.txt`

## Training

How to start training (RGB vs IR)

Key flags
Where checkpoints are saved
Example:

python train.py --data_root ds/train/rgb_images
python train.py --use_ir --data_root ds/train/ir_images

Mention:
Training is self-supervised
No labels required

## Evaluation Pipeline

Describe what evaluation means in this repo:
Latent prediction loss
Cosine similarity / squared distance
Evaluation does not involve classification accuracy

Example:

make rgb_eval
make ir_eval

## Representation Analysis & Visualization (Important Section)

This is a key differentiator of your project — give it a dedicated section.

Explain:

Patch token extraction
PCA fitted once on tokens
PCA reused for visualization
Visualizations show latent semantics, not pixel color

Example:

make rgb_viz NUM_VIZ=10
make ir_viz NUM_VIZ=10

Explain briefly:
Why outputs are blocky
Why PCA is modality-specific (RGB vs IR)

## End-to-End Workflow

One concise, high-level flow:

Train → Eval → Dump Tokens → Fit PCA → Visualize


Optionally mention:

make rgb_all
make ir_all

## Notes & Design Decisions

Short bullets, e.g.:
PCA is fit offline (not during eval)
RGB and IR use separate PCAs
Visualization uses model input transforms, not raw images
No supervised fine-tuning included
This prevents confusion.

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
