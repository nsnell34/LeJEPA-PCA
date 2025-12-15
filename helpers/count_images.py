import os

def count_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    count = 0

    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory to count images in")
    args = parser.parse_args()

    total = count_images(args.dir)
    print(f"Total images in {args.dir}: {total}")
