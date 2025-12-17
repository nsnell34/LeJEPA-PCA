import numpy as np
from sklearn.decomposition import PCA

def compute_global_pca(npy_paths, n_components=3):
    arrays = [np.load(p) for p in npy_paths]
    feats = np.concatenate(arrays, axis=0)
    print(f"Global PCA fitting on {feats.shape[0]} tokens with dimension {feats.shape[1]}")
    pca = PCA(n_components=n_components)
    pca.fit(feats)
    return pca


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--components", type=int, default=3)
    args = parser.parse_args()

    compute_global_pca(
        args.inputs,
        n_components=args.components,
        out_path=args.out
    )