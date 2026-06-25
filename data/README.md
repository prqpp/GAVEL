# Data

This directory holds (or is expected to hold) all CSV manifests that the
GAVEL pipeline consumes.  No image content is shipped; only file paths and
labels.

## Expected files

| Path | Schema | Source |
|---|---|---|
| `pairs_split{1..6}.csv` | `img0,img1` | Six disjoint random splits used for the statistical-significance analysis (rebuttal commitment to tUiR-W2). |
| `human_score.csv` | `img0,img1,human` | Mean-Opinion-Score across 28 annotators on the Human-Preference dev set (rebuttal qL7P-Q5 / tUiR-Q4). |
| `train_pairs.csv` | `img0,img1,label[,s_clip,s_vlm]` | Training set for the fusion head; the optional last two columns let `train_fusion.py` skip the slow CLIP+VLM forward passes. |

## Hard-Negative Pool construction

Following the rebuttal to BSmU-Q2, the hard-negative pool was built by a
fully automated, VLM-free pipeline that does **not** introduce human
selection bias:

1.  Aggregate source images from COCO-2017, Places365, Oxford-IIIT Pet, and
    Stanford Dogs.
2.  Compute HSV colour histograms and Gabor texture filters for every image.
3.  For every image, retrieve its `top-K` nearest neighbours by cosine
    similarity in the (HSV ⊕ Gabor) space.
4.  Drop pairs whose ground-truth metadata labels overlap.

This guarantees pairs are visually similar but semantically different by
construction, with no chance of cherry-picking.

## Sanity-check script

```bash
python - <<'PY'
import pandas as pd, hashlib, glob
for p in sorted(glob.glob("data/pairs_split*.csv")):
    df = pd.read_csv(p)
    h = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:10]
    print(f"{p:30s}  rows={len(df):4d}  md5={h}")
PY
```
