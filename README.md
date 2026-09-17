# Rooftop Fixture Candidate Verification

| | |
| --- | --- |
| Final rank | #11 |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 1 |
| Last submission | 2026-07-11 |

## Problem statement

### Overview

Flat-roof inspections often start from aerial imagery where drainage and ventilation fixtures occupy only a few pixels. Automated detectors can propose many candidate tiles, but an inspection workflow still needs to verify whether each tile is truly centered on a fixture, merely close to one, or just background clutter.

This challenge asks you to classify each proposed rooftop tile into one of three ordinal candidate classes. Strong solutions should use the image crop inside each candidate box, local roof texture, edge and contrast cues, and candidate geometry to separate exact fixture-centered tiles from visually similar seams, shadows, skylights, rooftop equipment, and off-center near misses. The held-out images emphasize lower-contrast, more strongly resampled roof crops.

### Dataset

### File descriptions

- `train.csv` -- 33,734 candidate tile rows for 260 labeled rooftop images, including the target `relevance` class.
- `test.csv` -- 19,636 candidate tile rows for 160 held-out rooftop images, with the same columns as `train.csv` except `relevance`.
- `sample_submission.csv` -- Submission template for every candidate tile in `test.csv`, filled with random class labels.
- `train/` -- Public JPEG rooftop images referenced by `image_path` in `train.csv`.
- `test/` -- Public JPEG rooftop images referenced by `image_path` in `test.csv`.

### Column descriptions

- `image_id` (string) -- Unique 12-character identifier for a rooftop image.
- `candidate_id` (string) -- Unique 12-character identifier for one candidate tile within an image.
- `image_path` (string) -- Relative path under `dataset/public/` for the JPEG image.
- `image_width` (int) -- Width of the image in pixels.
- `image_height` (int) -- Height of the image in pixels.
- `x_min` (int) -- Left pixel coordinate of the candidate tile.
- `y_min` (int) -- Top pixel coordinate of the candidate tile.
- `x_max` (int) -- Right pixel coordinate of the candidate tile.
- `y_max` (int) -- Bottom pixel coordinate of the candidate tile.
- `relevance` (float) -- Training target only. Exact fixture-centered tiles have class `3.0`, near-miss fixture-adjacent tiles have class `1.0`, and background tiles have class `0.0`.

### Evaluation

Submissions are scored with macro F1 across the three classes `0.0`, `1.0`, and `3.0`. Higher is better. Each class contributes equally, so predicting only the majority near-miss class will leave a low score even if many rows are labeled correctly.

```
import numpy as np

LABELS = [0.0, 1.0, 3.0]

def macro_f1(y_true, y_pred):

    scores = []

    for label in LABELS:

        tp = float(((y_true == label) & (y_pred == label)).sum())

        fp = float(((y_true != label) & (y_pred == label)).sum())

        fn = float(((y_true == label) & (y_pred != label)).sum())

        denom = 2.0 * tp + fp + fn

        scores.append(0.0 if denom == 0.0 else (2.0 * tp) / denom)

    return float(np.mean(scores))

def evaluate(answers, submission):

    merged = answers.merge(submission, on=["image_id", "candidate_id"])

    return macro_f1(

        merged["relevance"].astype(float).to_numpy(),

        merged["predicted_relevance"].astype(float).to_numpy(),

    )
```

### Submission

Submit a CSV file with one predicted class for every row in `test.csv`.

- `image_id` (string) -- The image identifier from `test.csv`.
- `candidate_id` (string) -- The candidate tile identifier from `test.csv`.
- `predicted_relevance` (float) -- Predicted class label. Must be exactly one of `0.0`, `1.0`, or `3.0`.

Example:

```
image_id,candidate_id,predicted_relevance

0156f5e50950,010a92b67bd5,1.0

0156f5e50950,02fb61976056,3.0
```

### Requirements

- The file must contain exactly 19,636 rows plus the header.
- Every `image_id` and `candidate_id` pair from `test.csv` must be present exactly once.
- Predicted labels must be finite numeric values and exactly one of `0.0`, `1.0`, or `3.0`.
- File format must be `.csv` with exact column names `image_id,candidate_id,predicted_relevance`.

### What Not To Use

- Do not use external copies of published annotation files, masks, or converted detection labels to recover held-out classes.
- Do not reverse-map hashed image or candidate identifiers back to original filenames or original annotation rows.
- Do not predict labels from cached candidate-generation recipes or hidden label maps. The intended task is to learn visual and geometric candidate verification from the public labeled examples.
