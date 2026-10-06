# Calibration sample (PaddleOCR Experience Center)

Labelled images used to fit the confidence calibrator (Module 3) that turns PaddleOCR's
raw line confidence into "probability this line was read correctly".

## Contents

- `sample_001.png` ... `sample_040.png`: 40 synthetic images, 2-4 printed lines each
  (118 lines), in several fonts and sizes. They are degraded in graded steps (blur,
  noise, faded ink, low resolution, slight rotation, JPEG artefacts) so that PaddleOCR
  misreads some lines. Calibration needs both correct and incorrect lines.
- `labels.csv`: ground truth, one row per image.

Regenerate (seeded, so the output is the same on a machine with the same fonts):

```bash
venv/Scripts/python -m app.calibration.make_sample
```

## labels.csv format

```csv
image,text
sample_001.png,"First line of the image
Second line of the image"
```

- `image`: path of the image, relative to the folder holding `labels.csv`.
- `text`: the ground-truth lines of that image, separated by newlines (quote the cell).
  A literal `\n` also works as a separator, for spreadsheets that cannot hold newlines.
  Line order does not matter: each OCR line is paired with its closest ground-truth line.

## Fitting the calibrator

```bash
venv/Scripts/python -m app.calibration.calibrate                       # this sample
venv/Scripts/python -m app.calibration.calibrate path/to/my_dataset    # your own data
venv/Scripts/python -m app.calibration.calibrate my.csv --match exact  # or a CSV path
```

Options: `--match cer` (default; a line is correct when its character error rate is at most
`--cer-threshold`, default `CALIBRATION_CER_THRESHOLD` = 0.1) or `--match exact`;
`--out DIR` (default `data/paddle_calibration/`); `--lang en`.

The Experience Center can do the same from the browser (`POST /api/calibrate` with a .zip
of `labels.csv` + images, or with the bundled sample).

Output: `calibrator.json` (used for every new result), `report.json` (metrics before and
after, measured on a held-out 30% of the lines) and `reliability_diagram.png`.

## Using your own data

Real scans of the documents you will process give a better calibrator than this synthetic
sample. Collect at least 30 images (ideally 100+ lines) with a mix of clean and hard pages,
type their text into `labels.csv`, and run the command above on the folder. The fit fails
with a clear message if fewer than 10 lines are read or if every line is correct (or every
line wrong).
