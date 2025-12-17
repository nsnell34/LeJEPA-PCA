PYTHON=python3

IR_CKPT=ckpts/lejepa_ir.pth
RGB_CKPT=ckpts/lejepa_rgb.pth

RGB_DATA_ROOT=ds/val/rgb_images
IR_DATA_ROOT=ds/val/ir_images

RGB_TOKENS=tokens/tokens_rgb.npy
IR_TOKENS=tokens/tokens_ir.npy

RGB_PCA=pca/jepa_pca_rgb.joblib
IR_PCA=pca/jepa_pca_ir.joblib

RGB_VIZ_OUT=viz/rgb
IR_VIZ_OUT=viz/ir

NUM_VIZ ?= 10

# -------- EVAL --------
rgb_eval:
	$(PYTHON) -m eval.lejepa_eval --val_root $(RGB_DATA_ROOT)

ir_eval:
	$(PYTHON) -m eval.lejepa_eval --val_root $(IR_DATA_ROOT) --use_ir


# -------- TOKENS --------

$(RGB_TOKENS):
	$(PYTHON) -m pca.lejepa_patch_tokens \
		--ckpt $(RGB_CKPT) \
		--root $(RGB_DATA_ROOT) \
		--out $(RGB_TOKENS)

$(IR_TOKENS):
	$(PYTHON) -m pca.lejepa_patch_tokens \
		--ckpt $(IR_CKPT) \
		--root $(IR_DATA_ROOT) \
		--out $(IR_TOKENS) \
		--use_ir


# -------- PCA --------

$(RGB_PCA): $(RGB_TOKENS)
	$(PYTHON) -m pca.global_pca \
		--inputs $(RGB_TOKENS) \
		--out $(RGB_PCA)

$(IR_PCA): $(IR_TOKENS)
	$(PYTHON) -m pca.global_pca \
		--inputs $(IR_TOKENS) \
		--out $(IR_PCA)


# -------- VISUALIZATION --------

rgb_viz: $(RGB_PCA)
	$(PYTHON) -m pca.batch_visualize \
		--ckpt $(RGB_CKPT) \
		--pca $(RGB_PCA) \
		--root $(RGB_DATA_ROOT) \
		--out $(RGB_VIZ_OUT) \
		--num $(NUM_VIZ)

ir_viz: $(IR_PCA)
	$(PYTHON) -m pca.batch_visualize \
		--ckpt $(IR_CKPT) \
		--pca $(IR_PCA) \
		--root $(IR_DATA_ROOT) \
		--out $(IR_VIZ_OUT) \
		--num $(NUM_VIZ) \
		--use_ir


# -------- PIPELINES --------

rgb_all: rgb_eval rgb_viz
ir_all: ir_eval ir_viz

.PHONY: rgb_eval ir_eval rgb_viz ir_viz rgb_all ir_all
