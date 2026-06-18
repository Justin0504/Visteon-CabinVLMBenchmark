"""Image quality gate for the exterior benchmark dataset.

Scans every unique image referenced by the training sharegpt files and flags:
  - low_res        : min(side) < 200 px
  - blurry         : variance-of-Laplacian below threshold (out-of-focus / motion blur)
  - overexposed    : mean brightness too high or >20% saturated (clipped white) pixels
  - underexposed   : mean brightness too low or >40% near-black pixels
  - near-duplicate : perceptual dHash within Hamming distance 5 of an earlier image

Outputs data/quality_report.json (summary + per-image records + flagged list).
CPU-only, no GPU. Metrics computed on a 256px resize for speed/consistency;
the low_res flag uses the ORIGINAL resolution.
"""
import json, glob
import numpy as np
from PIL import Image

R = "/data/haiyuez/visteon_cabin_vlm"
BLUR_T = 60.0          # variance-of-Laplacian threshold (tunable)
SAT_T, DARK_T = 20.0, 40.0
HAM_T = 5              # near-duplicate Hamming distance

# 1) collect unique images from all training sharegpt files
imgs = set()
for f in glob.glob(R + "/data/*_sharegpt.json"):
    try:
        for x in json.load(open(f)):
            for im in x.get("images", []):
                imgs.add(im)
    except Exception:
        pass
imgs = sorted(imgs)
print("UNIQUE_IMAGES", len(imgs), flush=True)

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def lap_var(g):
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(g, (3, 3))
    return float((w * LAP).sum(axis=(-1, -2)).var())


def dhash(im, h=8):
    g = np.asarray(im.convert("L").resize((h + 1, h)), dtype=np.int16)
    diff = g[:, 1:] > g[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def ham(a, b):
    return bin(a ^ b).count("1")


report = []
for i, p in enumerate(imgs):
    try:
        im = Image.open(p).convert("RGB")
    except Exception:
        report.append({"image": p, "error": "open_fail", "flags": ["open_fail"]})
        continue
    W, H = im.size
    small = im.resize((256, 256))
    g = np.asarray(small.convert("L"), dtype=np.float32)
    blur = lap_var(g)
    mean = float(g.mean()); sat = float((g >= 250).mean() * 100); dark = float((g <= 5).mean() * 100)
    flags = []
    if min(W, H) < 200: flags.append("low_res")
    if blur < BLUR_T: flags.append("blurry")
    if mean > 220 or sat > SAT_T: flags.append("overexposed")
    if mean < 35 or dark > DARK_T: flags.append("underexposed")
    report.append({"image": p, "w": W, "h": H, "blur": round(blur, 1),
                   "mean": round(mean, 1), "sat%": round(sat, 1), "flags": flags, "dhash": dhash(im)})
    if (i + 1) % 1000 == 0:
        print("scanned", i + 1, flush=True)

# 2) near-duplicate clustering by dHash
clusters = []
dups = 0
for r in report:
    if "dhash" not in r:
        continue
    placed = False
    for c in clusters:
        if ham(r["dhash"], c) <= HAM_T:
            r["dup"] = True; dups += 1; placed = True; break
    if not placed:
        clusters.append(r["dhash"])

# 3) summary
from collections import Counter
fc = Counter()
for r in report:
    for fl in r.get("flags", []):
        fc[fl] += 1
summary = {"total": len(report),
           "clean": sum(1 for r in report if not r.get("flags") and not r.get("dup")),
           "flagged_any": sum(1 for r in report if r.get("flags")),
           "by_flag": dict(fc),
           "near_duplicates": dups,
           "unique_after_dedup": len(clusters)}
json.dump({"summary": summary, "records": report},
          open(R + "/data/quality_report.json", "w"))
print("QG_SUMMARY", json.dumps(summary, ensure_ascii=False))
print("QG_DONE")
