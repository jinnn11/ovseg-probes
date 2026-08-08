"""Download candidate images for all probe files.

Collects the union of image_ids across all probe, control, and extras
files, splits by source (COCO vs Visual Genome), downloads with
concurrent workers, retries with backoff, and verifies each file
opens with PIL.

Usage:
    python -m src.download_images
    python -m src.download_images --workers 16
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from src.schema import load_probes

# ── Paths ──────────────────────────────────────────────────────────
PROBES_DIR = Path("probes")
VG_IMGDATA_PATH = Path("data/vg/image_data.json")
IMAGES_DIR = Path("data/images")
FAILURE_LOG = Path("data/images/download_failures.log")

COCO_URL_TEMPLATE = "http://images.cocodataset.org/train2017/{:012d}.jpg"

MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds
TIMEOUT = 30  # seconds per request


# ── Collect image IDs ──────────────────────────────────────────────

def _collect_image_ids() -> tuple[set[int], set[int]]:
    """Return (coco_ids, vg_ids) from all probe files."""
    coco_ids: set[int] = set()
    vg_ids: set[int] = set()

    probe_files = sorted(PROBES_DIR.glob("*.json"))
    for pf in probe_files:
        try:
            probes = load_probes(pf)
        except Exception as e:
            print(f"  Skipping {pf.name}: {e}")
            continue
        for p in probes:
            if p.image_source == "coco_train2017":
                coco_ids.add(int(p.image_id))
            elif p.image_source == "visual_genome":
                vg_ids.add(int(p.image_id))
            elif p.image_source == "lvis_v1_train":
                coco_ids.add(int(p.image_id))

    return coco_ids, vg_ids


def _load_vg_urls(vg_ids: set[int]) -> dict[int, str]:
    """Load VG image URLs for the requested IDs."""
    with VG_IMGDATA_PATH.open() as f:
        data = json.load(f)
    return {
        img["image_id"]: img["url"]
        for img in data
        if img["image_id"] in vg_ids
    }


# ── Download ───────────────────────────────────────────────────────

def _download_one(url: str, dest: Path) -> str | None:
    """Download url to dest with retries. Return error string or None."""
    if dest.exists() and dest.stat().st_size > 0:
        if _verify_image(dest):
            return None

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            req = Request(url, headers={"User-Agent": "ovseg-probes/1.0"})
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            dest.write_bytes(data)
            if _verify_image(dest):
                return None
            else:
                return f"corrupt image: {url}"
        except (URLError, HTTPError, OSError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE ** (attempt + 1))
            else:
                return f"{type(e).__name__}: {e} — {url}"

    return f"max retries exceeded: {url}"


def _verify_image(path: Path) -> bool:
    """Check that PIL can open the file."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_all(workers: int = 12) -> None:
    """Download all candidate images."""
    print("Collecting image IDs from probe files …")
    coco_ids, vg_ids = _collect_image_ids()
    print(f"  COCO/LVIS: {len(coco_ids)} images")
    print(f"  VG:        {len(vg_ids)} images")

    # Build download tasks: (url, destination_path)
    tasks: list[tuple[str, Path]] = []

    coco_dir = IMAGES_DIR / "coco"
    for cid in sorted(coco_ids):
        url = COCO_URL_TEMPLATE.format(cid)
        dest = coco_dir / f"{cid:012d}.jpg"
        tasks.append((url, dest))

    vg_dir = IMAGES_DIR / "vg"
    if vg_ids:
        vg_urls = _load_vg_urls(vg_ids)
        missing_urls = vg_ids - set(vg_urls.keys())
        if missing_urls:
            print(f"  Warning: {len(missing_urls)} VG image IDs have no URL")
        for vid in sorted(vg_urls.keys()):
            url = vg_urls[vid]
            ext = url.rsplit(".", 1)[-1] if "." in url else "jpg"
            dest = vg_dir / f"{vid}.{ext}"
            tasks.append((url, dest))

    # Check how many already exist
    already = sum(1 for _, dest in tasks if dest.exists() and dest.stat().st_size > 0)
    print(f"\nTotal tasks: {len(tasks)} ({already} already downloaded)")

    failures: list[str] = []
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, url, dest): (url, dest)
            for url, dest in tasks
        }
        for future in as_completed(futures):
            completed += 1
            err = future.result()
            if err:
                failures.append(err)
            if completed % 100 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{len(tasks)}] "
                      f"{elapsed:.0f}s ({rate:.1f} img/s) "
                      f"failures={len(failures)}")

    # Log failures
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        with FAILURE_LOG.open("w") as f:
            for line in failures:
                f.write(line + "\n")
        print(f"\n{len(failures)} failures logged → {FAILURE_LOG}")
    else:
        if FAILURE_LOG.exists():
            FAILURE_LOG.unlink()
        print("\nAll images downloaded and verified successfully.")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. "
          f"{len(tasks) - len(failures)}/{len(tasks)} images OK.")


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of concurrent download workers")
    args = parser.parse_args()
    download_all(workers=args.workers)


if __name__ == "__main__":
    main()
