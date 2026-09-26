# Recognition with docTR OCR + new HTR line pipeline

Recognition-only run (`run_eval --skip-qa`) on a laptop without Tesseract, so `OCR_ENGINE=auto` used docTR; HTR used OpenCV rule removal + docTR line detection. Compare with `report.md` (Tesseract + projection splitter). Synthetic 'handwriting' is computer fonts, which docTR reads well, so those pages were routed to OCR (see `format_detection.detect_format`); real handwriting still needs a real-sample test.

## 1. Recognition (Module 2)

| Format | Pages | Detection accuracy | Mean CER | Mean WER | Mean raw conf | Mean calibrated conf |
|---|---|---|---|---|---|---|
| typed | 4 | 100% | 0.000 | 0.000 | 1.00 | 1.00 |
| scanned | 14 | 100% | 0.005 | 0.034 | 0.95 | 0.98 |
| handwritten | 16 | 0% | 0.017 | 0.075 | 0.93 | 0.95 |

Per document:

| Document | Format | Detected as | CER | WER | Raw conf | s/page |
|---|---|---|---|---|---|---|
| dbms_notes | typed | typed | 0.000 | 0.000 | 1.00 | 0.0 |
| os_scheduling | typed | typed | 0.000 | 0.000 | 1.00 | 0.0 |
| computer_networks | scanned | scanned | 0.000 | 0.000 | 0.95 | 7.6 |
| machine_learning_photocopy | scanned | scanned | 0.012 | 0.081 | 0.88 | 8.2 |
| biology_notes | handwritten | scanned | 0.000 | 0.000 | 0.93 | 6.8 |
| physics_rough_notes | handwritten | scanned | 0.076 | 0.310 | 0.76 | 7.3 |
| cal_scanned_0 | scanned | scanned | 0.002 | 0.029 | 0.97 | 16.1 |
| cal_scanned_1 | scanned | scanned | 0.002 | 0.029 | 0.97 | 12.1 |
| cal_scanned_2 | scanned | scanned | 0.007 | 0.042 | 0.96 | 12.6 |
| cal_scanned_3 | scanned | scanned | 0.000 | 0.000 | 0.95 | 9.2 |
| cal_scanned_4 | scanned | scanned | 0.009 | 0.056 | 0.95 | 7.6 |
| cal_handwritten_0 | handwritten | scanned | 0.000 | 0.000 | 0.97 | 12.5 |
| cal_handwritten_1 | handwritten | scanned | 0.005 | 0.026 | 0.98 | 8.7 |
| cal_handwritten_2 | handwritten | scanned | 0.024 | 0.111 | 0.95 | 9.7 |
| cal_handwritten_3 | handwritten | scanned | 0.009 | 0.053 | 0.95 | 9.6 |
| cal_handwritten_4 | handwritten | scanned | 0.005 | 0.055 | 0.97 | 8.6 |
| cal_handwritten_5 | handwritten | scanned | 0.014 | 0.042 | 0.93 | 9.0 |

