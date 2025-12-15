import os
import shutil
import uuid

def collect_images(folder, exts):
    img_paths = []
    for dp, dirs, fs in os.walk(folder):
        dirs.sort()
        fs.sort()
        for f in fs:
            if f.lower().endswith(exts):
                img_paths.append(os.path.join(dp, f))
    return img_paths


def sort_pca_in_place(input_dir, pca_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    print("Reading input images...")
    input_imgs = collect_images(input_dir, exts)
    input_names = [os.path.basename(p) for p in input_imgs]

    print("Reading PCA images...")
    pca_imgs = collect_images(pca_dir, exts)
    pca_lookup = {os.path.basename(p): p for p in pca_imgs}

    print(f"Input count: {len(input_imgs)}")
    print(f"PCA count:   {len(pca_imgs)}")

    temp_names = {}

    print("\nRenaming PCA files to temporary names...")
    for fname, full_path in pca_lookup.items():
        temp_name = f"{uuid.uuid4().hex}.tmp"
        temp_path = os.path.join(pca_dir, temp_name)
        os.rename(full_path, temp_path)
        temp_names[fname] = temp_path

    kept = 0
    missing = 0

    print("Sorting PCA folder to match input order...\n")

    for fname in input_names:
        if fname in temp_names:
            old = temp_names[fname]
            new = os.path.join(pca_dir, fname)
            os.rename(old, new)
            kept += 1
        else:
            missing += 1

    print("===== SUMMARY =====")
    print(f"Input images     : {len(input_imgs)}")
    print(f"PCA matched      : {kept}")
    print(f"PCA missing      : {missing}")
    print(f"PCA folder sorted in place: {pca_dir}")
    print("===================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of original RGB/IR images")
    parser.add_argument("--pca", required=True, help="Directory containing PCA outputs to sort in place")

    args = parser.parse_args()
    
    sort_pca_in_place(args.input, args.pca)
