from pathlib import Path
import csv
import argparse
from collections import defaultdict

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_ids_from_annotations(csv_path: Path):
    ids = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            sample_id = str(row[0]).strip().lstrip("\ufeff")
            if sample_id:
                ids.append(sample_id)
    return ids


def find_existing_image(images_dir: Path, sample_id: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{sample_id}{ext}"
        if p.exists():
            return p
    return None


def check_split(split_dir: Path):
    annotations_path = split_dir / "annotations.csv"
    images_dir = split_dir / "images"
    texts_dir = split_dir / "texts"

    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not texts_dir.exists():
        raise FileNotFoundError(f"Missing texts dir: {texts_dir}")

    sample_ids = read_ids_from_annotations(annotations_path)

    missing_image = []
    missing_text = []
    missing_both = []

    for sid in sample_ids:
        img_path = find_existing_image(images_dir, sid)
        txt_path = texts_dir / f"{sid}.txt"

        img_missing = img_path is None
        txt_missing = not txt_path.exists()

        if img_missing:
            missing_image.append(sid)
        if txt_missing:
            missing_text.append(sid)
        if img_missing and txt_missing:
            missing_both.append(sid)

    return {
        "total": len(sample_ids),
        "missing_image": missing_image,
        "missing_text": missing_text,
        "missing_both": missing_both,
    }


def write_missing_csv(out_csv: Path, split_results):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "sample_id", "missing_image", "missing_text"])

        for split, r in split_results.items():
            mi = set(r["missing_image"])
            mt = set(r["missing_text"])
            all_ids = sorted(mi | mt, key=lambda x: (len(x), x) if x.isdigit() else (10**9, x))
            for sid in all_ids:
                writer.writerow([
                    split,
                    sid,
                    int(sid in mi),
                    int(sid in mt),
                ])


def main():
    parser = argparse.ArgumentParser(description="Check missing image/text files in multimodal dataset")
    parser.add_argument("--dataset-root", type=str, default=r"D:\\data\\Twitter", help="Dataset root path")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to check")
    parser.add_argument("--out-csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    split_results = {}
    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] split dir not found, skip: {split_dir}")
            continue

        r = check_split(split_dir)
        split_results[split] = r

        print(f"[{split}] total={r['total']}")
        print(
            f"  missing_image={len(r['missing_image'])}, "
            f"missing_text={len(r['missing_text'])}, "
            f"missing_both={len(r['missing_both'])}"
        )

    if not split_results:
        print("No valid split checked.")
        return

    out_csv = Path(args.out_csv) if args.out_csv else dataset_root / "missing_modalities_report.csv"
    write_missing_csv(out_csv, split_results)
    print(f"\nSaved missing report to: {out_csv}")


if __name__ == "__main__":
    main()
