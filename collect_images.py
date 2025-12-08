from pathlib import Path
import shutil

base_path = Path("/home/megrad/thor_pool/Raleigh_Skydio_Collect/scans")
rgb_out = Path("/home/megrad/Documents/Github/lejepa/ds/dummy/rgb")
rgb_out.mkdir(parents=True, exist_ok=True)

RGB_EXTS = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}


def resolve_rgb_dir(scan_path: Path) -> Path | None:

    fg_candidates = [
        scan_path / "Final_Gcrop" / "img",
        scan_path / "final_gcrop" / "img",
    ]
    for d in fg_candidates:
        if d.exists() and d.is_dir():
            return d

    coarse_candidates = [
        scan_path / "coarse_IR" / "img",
        scan_path / "coarse_ir" / "img",
    ]
    for d in coarse_candidates:
        if d.exists() and d.is_dir():
            return d

    return None


scan_paths = [p for p in base_path.iterdir() if p.is_dir()]
print(f"Found {len(scan_paths)} scan dirs under {base_path}")

num_rgb = 0

for scan_path in scan_paths:
    scan_name = scan_path.name
    print(f"\n[SCAN] {scan_name}")

    rgb_dir = resolve_rgb_dir(scan_path)
    print(f"  Selected RGB dir: {rgb_dir}")

    if rgb_dir is None:
        print("  [WARN] No Final_Gcrop or coarse_IR img/image dir found, skipping.")
        continue

    img_files = sorted(
        [p for p in rgb_dir.iterdir() if p.is_file() and p.suffix in RGB_EXTS]
    )
    print(f"  RGB images found: {len(img_files)}")

    for img_path in img_files:
        out_stem = f"{scan_name}_{img_path.stem}"
        out_name = f"{out_stem}{img_path.suffix}"
        dest_path = rgb_out / out_name

        shutil.copy2(img_path, dest_path)
        num_rgb += 1

print(f"\nDone. Copied {num_rgb} RGB images into {rgb_out}")
