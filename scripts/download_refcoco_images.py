"""Download COCO images needed for RefCOCO evaluation."""
import json, os, requests
from pathlib import Path

refcoco = json.load(open("data/refcoco/refcoco_val.json"))
image_ids = list(dict.fromkeys(s["image_id"] for s in refcoco[:500]))
out_dir = Path("data/images/coco")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Need {len(image_ids)} unique images for first 500 RefCOCO samples")
downloaded = 0
skipped = 0

for i, img_id in enumerate(image_ids):
    fname = f"{img_id:012d}.jpg"
    out_path = out_dir / fname
    if out_path.exists():
        skipped += 1
        continue
    url = f"http://images.cocodataset.org/train2014/COCO_train2014_{fname}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            downloaded += 1
        else:
            url2 = f"http://images.cocodataset.org/val2014/COCO_val2014_{fname}"
            r2 = requests.get(url2, timeout=15)
            if r2.status_code == 200:
                out_path.write_bytes(r2.content)
                downloaded += 1
            else:
                print(f"  MISS: {img_id}")
    except Exception as e:
        print(f"  ERROR {img_id}: {e}")
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(image_ids)}] downloaded={downloaded} skipped={skipped}")

print(f"Done: downloaded={downloaded}, already_existed={skipped}, total={len(image_ids)}")
