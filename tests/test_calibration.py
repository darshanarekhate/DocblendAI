"""Module 3 (Experience Center): calibrators, metrics, labelling, selection, saving and loading.

Never loads PaddleOCR: run_calibration is driven by a fake ocr_fn.
"""

import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.calibration import (
    Calibrator,
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    active_method,
    calibrate_confidences,
    clear_active_cache,
    fit_and_select,
    load_active,
)
from app.calibration import calibrate as calibrate_module
from app.calibration.calibrate import run_calibration
from app.calibration.labeling import cer, label_lines, levenshtein, normalize, split_truth
from app.calibration.make_sample import generate
from app.calibration.metrics import brier, ece, mce, nll, reliability_bins
from app.calibration.selection import stratified_split
from app.config import settings

ALL = [TemperatureScaler, PlattScaler, IsotonicCalibrator]
REPORT_KEYS = {
    "method", "selected_by", "fitted_at", "dataset", "match", "cer_threshold", "n_lines", "n_correct",
    "n_train", "n_val", "candidates", "metrics", "bins", "diagram",
}
BIN_KEYS = {"lo", "hi", "count", "mean_confidence", "accuracy"}
METRIC_KEYS = {"ece", "mce", "brier", "nll"}


@pytest.fixture(autouse=True)
def calibration_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the active calibrator at a throwaway folder, never data/paddle_calibration."""
    folder = tmp_path / "paddle_calibration"
    monkeypatch.setattr(settings, "paddle_calibration_dir", folder)
    clear_active_cache()
    yield folder
    clear_active_cache()


def overconfident(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raw confidences sharper than the truth: true p = sigmoid(z), reported = sigmoid(3z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(1.0, 1.2, n)
    labels = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return 1 / (1 + np.exp(-3 * z)), labels


# ---------- calibrators ----------


@pytest.mark.parametrize("cls", ALL)
def test_calibrators_reduce_ece_on_overconfident_data(cls):
    conf, labels = overconfident(4000)
    model = cls().fit(conf[:2000], labels[:2000])
    raw, calibrated = ece(conf[2000:], labels[2000:]), ece(model.predict(conf[2000:]), labels[2000:])
    assert calibrated < raw / 2
    assert brier(model.predict(conf[2000:]), labels[2000:]) < brier(conf[2000:], labels[2000:])


def test_temperature_recovers_known_temperature():
    conf, labels = overconfident(20000, seed=1)
    assert TemperatureScaler().fit(conf, labels).temperature == pytest.approx(3.0, rel=0.1)


def test_platt_recovers_known_slope_and_offset():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, 20000)
    labels = (rng.random(z.size) < 1 / (1 + np.exp(-(0.5 * z - 1.0)))).astype(int)
    model = PlattScaler().fit(1 / (1 + np.exp(-z)), labels)
    assert model.a == pytest.approx(0.5, abs=0.05)
    assert model.b == pytest.approx(-1.0, abs=0.08)


def test_isotonic_pools_violators_and_is_monotone():
    model = IsotonicCalibrator(prior=0).fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])  # pure PAV
    assert model.predict([0.1, 0.2, 0.3, 0.4]).tolist() == pytest.approx([1e-6, 0.5, 0.5, 1 - 1e-6])
    conf, labels = overconfident(3000, seed=3)
    model = IsotonicCalibrator().fit(conf, labels)
    grid = model.predict(np.linspace(0, 1, 1001))
    assert np.all(np.diff(grid) >= -1e-12)
    assert grid.min() >= 1e-6 and grid.max() <= 1 - 1e-6


def test_isotonic_merges_equal_confidences():
    model = IsotonicCalibrator(prior=0).fit([0.5, 0.5, 0.5, 0.9], [0, 1, 1, 1])
    assert model.predict([0.5])[0] == pytest.approx(2 / 3)


def test_isotonic_prior_keeps_small_blocks_off_zero_and_one():
    # 12 misreads below 0.8, 30 correct reads above: pure PAV says exactly 0 and 1.
    conf = [0.5 + 0.02 * i for i in range(12)] + [0.9 + 0.003 * i for i in range(30)]
    labels = [0] * 12 + [1] * 30
    model = IsotonicCalibrator().fit(conf, labels)
    low, high = model.predict([0.55, 0.95])
    assert low == pytest.approx(1 / 14) and high == pytest.approx(31 / 32)
    restored = Calibrator.from_dict(model.to_dict())
    assert restored.prior == 1.0 and restored.predict([0.55])[0] == pytest.approx(low)


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize(
    "conf, labels",
    [
        ([0.9, 0.8, 0.95, 0.7], [1, 1, 1, 1]),  # one class (correct)
        ([0.9, 0.8, 0.95, 0.7], [0, 0, 0, 0]),  # one class (incorrect)
        ([0.6, 0.6, 0.6, 0.6], [1, 0, 1, 1]),  # constant confidence
        ([0.7], [1]),  # a single line
        ([0.0, 1.0, 1.0, 0.0, 1.0], [0, 1, 0, 1, 1]),  # exactly 0 and 1
    ],
)
def test_calibrators_survive_edge_cases(cls, conf, labels):
    out = cls().fit(conf, labels).predict([0.0, 0.3, 0.6, 1.0])
    assert out.shape == (4,)
    assert np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1))


def test_constant_confidence_maps_to_observed_accuracy():
    for cls in (PlattScaler, IsotonicCalibrator):
        out = cls().fit([0.6] * 8, [1, 1, 1, 1, 1, 1, 0, 0]).predict([0.6])[0]
        assert out == pytest.approx(0.75, abs=0.06)


@pytest.mark.parametrize("cls", ALL)
def test_fit_rejects_bad_input(cls):
    with pytest.raises(ValueError):
        cls().fit([], [])
    with pytest.raises(ValueError):
        cls().fit([0.5, 0.6], [1])
    with pytest.raises(ValueError):
        cls().fit([0.5, 0.6], [1, 2])


@pytest.mark.parametrize("cls", ALL)
def test_serialisation_round_trip(cls):
    conf, labels = overconfident(500, seed=4)
    model = cls().fit(conf, labels)
    data = json.loads(json.dumps(model.to_dict()))
    assert data["method"] == cls.method
    restored = Calibrator.from_dict(data)
    assert type(restored) is cls
    probe = np.linspace(0, 1, 50)
    np.testing.assert_allclose(restored.predict(probe), model.predict(probe))


def test_from_dict_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown calibration method"):
        Calibrator.from_dict({"method": "magic"})
    with pytest.raises(ValueError):
        Calibrator.from_dict([1, 2])  # type: ignore[arg-type]


# ---------- metrics ----------

CONF = [0.15, 0.25, 0.25, 0.95]
LABELS = [0, 1, 0, 1]


def test_metrics_hand_computed():
    # Bins: [0.1,0.2) gap 0.15 (n=1); [0.2,0.3) acc 0.5 vs 0.25 (n=2); [0.9,1.0] gap 0.05 (n=1).
    assert ece(CONF, LABELS) == pytest.approx((0.15 + 2 * 0.25 + 0.05) / 4)
    assert mce(CONF, LABELS) == pytest.approx(0.25)
    assert brier(CONF, LABELS) == pytest.approx((0.0225 + 0.5625 + 0.0625 + 0.0025) / 4)
    expected_nll = -(math.log(0.85) + math.log(0.25) + math.log(0.75) + math.log(0.95)) / 4
    assert nll(CONF, LABELS) == pytest.approx(expected_nll)


def test_perfect_predictions_score_zero_and_nll_is_clipped():
    assert ece([1.0, 0.0], [1, 0]) == 0.0
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert nll([1.0], [0]) == pytest.approx(-math.log(1e-6))


def test_reliability_bins_shape():
    bins = reliability_bins(CONF + [1.0], LABELS + [1])
    assert len(bins) == 10
    assert all(set(b) == BIN_KEYS for b in bins)
    assert [b["lo"] for b in bins] == pytest.approx([k / 10 for k in range(10)])
    assert bins[2] == {"lo": 0.2, "hi": 0.3, "count": 2, "mean_confidence": 0.25, "accuracy": 0.5}
    assert bins[9]["count"] == 2  # 1.0 lands in the last bin
    assert bins[0] == {"lo": 0.0, "hi": 0.1, "count": 0, "mean_confidence": None, "accuracy": None}
    assert sum(b["count"] for b in bins) == 5


# ---------- labelling ----------


def test_normalize_levenshtein_and_cer():
    assert normalize("  Hello \t  world \n") == "Hello world"
    assert levenshtein("kitten", "sitting") == 3
    assert cer("Hel1o", "Hello") == pytest.approx(0.2)
    assert cer("", "") == 0.0 and cer("x", "") == 1.0
    assert split_truth("a\nb\r\n\n c ") == ["a", "b", "c"]
    assert split_truth("first\\nsecond") == ["first", "second"]


def test_label_lines_matches_out_of_order_and_flags_extras():
    truth = ["The quick brown fox", "jumps over the lazy dog"]
    ocr = [("jumps ovor the lazy dog", 0.9), ("noise", 0.4), ("The quick brown fox", 0.99)]
    labels = label_lines(ocr, truth, match="cer", cer_threshold=0.1)
    assert [l.truth for l in labels] == ["jumps over the lazy dog", None, "The quick brown fox"]
    assert [l.correct for l in labels] == [1, 0, 1]
    assert labels[0].cer == pytest.approx(1 / 23)
    exact = label_lines(ocr, truth, match="exact")
    assert [l.correct for l in exact] == [0, 0, 1]


def test_each_truth_line_is_used_once():
    labels = label_lines([("Hello world", 0.9), ("Hello world", 0.8)], ["Hello world"])
    assert [l.correct for l in labels] == [1, 0]
    assert label_lines([("anything", 0.5)], [])[0].correct == 0
    with pytest.raises(ValueError):
        label_lines([], [], match="fuzzy")


# ---------- selection ----------


def test_stratified_split_is_deterministic_and_keeps_both_classes():
    labels = np.array([1] * 20 + [0] * 5, dtype=float)
    train, val = stratified_split(labels, 0.3, seed=0)
    again = stratified_split(labels, 0.3, seed=0)
    assert train.tolist() == again[0].tolist() and val.tolist() == again[1].tolist()
    assert sorted(train.tolist() + val.tolist()) == list(range(25))
    assert set(labels[val]) == {0.0, 1.0} and set(labels[train]) == {0.0, 1.0}
    assert len(val) == 6 + 2


def test_fit_and_select_report_shape():
    conf, labels = overconfident(300, seed=5)
    calibrator, report = fit_and_select(conf, labels)
    assert report["method"] in ("temperature", "platt", "isotonic")
    assert calibrator.method == report["method"]
    assert report["selected_by"] == "validation ECE"
    assert report["n_lines"] == 300 and report["n_correct"] == int(labels.sum())
    assert report["n_train"] + report["n_val"] == 300 and report["n_val"] == 90
    assert set(report["candidates"]) == {"temperature", "platt", "isotonic"}
    assert all("val_ece" in c for c in report["candidates"].values())
    best = min(c["val_ece"] for c in report["candidates"].values())
    assert report["candidates"][report["method"]]["val_ece"] == best
    for when in ("before", "after"):
        assert set(report["metrics"][when]) == METRIC_KEYS
        assert len(report["bins"][when]) == 10 and all(set(b) == BIN_KEYS for b in report["bins"][when])
        assert sum(b["count"] for b in report["bins"][when]) == 90
    assert report["metrics"]["after"]["ece"] == report["candidates"][report["method"]]["val_ece"]
    assert report["metrics"]["after"]["ece"] < report["metrics"]["before"]["ece"]
    json.dumps(report)  # serialisable
    assert fit_and_select(conf, labels)[1] == report  # deterministic


def test_fit_and_select_rejects_too_little_data():
    with pytest.raises(ValueError):
        fit_and_select([0.5], [1])


# ---------- active calibrator ----------


def test_no_calibrator_means_uncalibrated(calibration_dir):
    assert load_active() is None
    assert active_method() is None
    assert calibrate_confidences([0.2, 0.95]) == ([0.2, 0.95], "uncalibrated")


def test_saved_calibrator_is_loaded_cached_and_applied(calibration_dir):
    assert load_active() is None  # cached "none"
    calibration_dir.mkdir(parents=True)
    (calibration_dir / "calibrator.json").write_text(json.dumps(PlattScaler(1.0, -1.0).to_dict()))
    assert load_active() is None  # still cached until cleared
    clear_active_cache()
    assert isinstance(load_active(), PlattScaler) and active_method() == "platt"
    values, status = calibrate_confidences([0.5, 0.9])
    assert status == "calibrated"
    assert values == pytest.approx([1 / (1 + math.e), 1 / (1 + math.exp(-(math.log(9) - 1)))])
    assert calibrate_confidences([]) == ([], "calibrated")


def test_corrupt_calibrator_is_ignored_with_warning(calibration_dir, caplog):
    calibration_dir.mkdir(parents=True)
    (calibration_dir / "calibrator.json").write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert load_active() is None
    assert "unreadable calibrator" in caplog.text
    assert calibrate_confidences([0.4]) == ([0.4], "uncalibrated")


# ---------- run_calibration ----------


def read_truth(dataset: Path) -> dict[str, list[str]]:
    with open(dataset / "labels.csv", encoding="utf-8", newline="") as f:
        return {row["image"]: split_truth(row["text"]) for row in csv.DictReader(f)}


def fake_ocr_for(dataset: Path):
    """Reads each image's ground truth; every third line is garbled with a lower confidence."""
    truth = read_truth(dataset)
    calls = []

    def ocr_fn(path: Path) -> list[tuple[str, float]]:
        calls.append(Path(path))
        lines = []
        for text in truth[Path(path).name]:
            k = len(calls) * 7 + len(lines)
            if k % 3 == 0:
                lines.append((text[::-1], 0.6 + 0.03 * (k % 5)))
            else:
                lines.append((text, 0.85 + 0.02 * (k % 7)))
        return lines

    return ocr_fn, calls


@pytest.fixture(scope="module")
def sample_dataset(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("calibration_sample")
    generate(folder, count=8)
    return folder


def test_generated_sample_is_valid(sample_dataset):
    truth = read_truth(sample_dataset)
    assert len(truth) == 8
    for name, lines in truth.items():
        assert 2 <= len(lines) <= 4
        with Image.open(sample_dataset / name) as image:
            assert image.mode == "L"


def test_run_calibration_end_to_end(sample_dataset, calibration_dir):
    ocr_fn, calls = fake_ocr_for(sample_dataset)
    seen = []
    report = run_calibration(sample_dataset, ocr_fn, progress=lambda done, total: seen.append((done, total)))

    assert len(calls) == 8 and seen[-1] == (8, 8) and len(seen) == 8
    assert set(report) == REPORT_KEYS
    assert report["match"] == "cer" and report["cer_threshold"] == settings.calibration_cer_threshold
    assert report["diagram"] == "reliability_diagram.png"
    assert 0 < report["n_correct"] < report["n_lines"]
    for name in ("calibrator.json", "report.json", "reliability_diagram.png"):
        assert (calibration_dir / name).is_file()
    assert json.loads((calibration_dir / "report.json").read_text()) == report
    with Image.open(calibration_dir / "reliability_diagram.png") as image:
        assert image.format == "PNG" and image.size == (900, 450)
        image.verify()

    active = load_active()  # cache was cleared by run_calibration
    assert active is not None and active.method == report["method"]
    assert calibrate_confidences([0.9])[1] == "calibrated"


def test_run_calibration_accepts_csv_path_and_exact_match(sample_dataset, tmp_path):
    ocr_fn, _ = fake_ocr_for(sample_dataset)
    out = tmp_path / "out"
    report = run_calibration(sample_dataset / "labels.csv", ocr_fn, out_dir=out, match="exact", cer_threshold=0.0)
    assert report["match"] == "exact" and report["cer_threshold"] == 0.0
    assert (out / "calibrator.json").is_file()


def test_run_calibration_rejects_unusable_data(sample_dataset, tmp_path):
    out = tmp_path / "out"
    truth = read_truth(sample_dataset)
    perfect = lambda path: [(t, 0.9) for t in truth[Path(path).name]]  # noqa: E731
    with pytest.raises(ValueError, match="both correct and incorrect"):
        run_calibration(sample_dataset, perfect, out_dir=out)
    with pytest.raises(ValueError, match="at least 10"):
        run_calibration(sample_dataset, lambda path: [("x", 0.5)], out_dir=out)
    with pytest.raises(ValueError, match="no labels file"):
        run_calibration(tmp_path / "missing", perfect, out_dir=out)
    with pytest.raises(ValueError, match="match"):
        run_calibration(sample_dataset, perfect, out_dir=out, match="fuzzy")
    assert not (out / "calibrator.json").exists()


def test_read_labels_reports_bad_csv(tmp_path):
    (tmp_path / "labels.csv").write_text("file,truth\na.png,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image,text"):
        run_calibration(tmp_path, lambda p: [], out_dir=tmp_path / "o")
    (tmp_path / "labels.csv").write_text("image,text\nmissing.png,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image not found: missing.png"):
        run_calibration(tmp_path, lambda p: [], out_dir=tmp_path / "o")


def test_cli_uses_recognize_lines(sample_dataset, tmp_path, monkeypatch, capsys):
    from app.modules import paddleocr_service as service_module

    ocr_fn, _ = fake_ocr_for(sample_dataset)
    langs = []

    def fake_recognize(path, lang="en"):
        langs.append(lang)
        return ocr_fn(path)

    monkeypatch.setattr(service_module.paddleocr_service, "recognize_lines", fake_recognize, raising=False)
    calibrate_module.main([str(sample_dataset), "--out", str(tmp_path / "cli"), "--lang", "en"])
    printed = capsys.readouterr().out
    assert "Chosen:" in printed and "ece" in printed
    assert set(langs) == {"en"}
    assert (tmp_path / "cli" / "report.json").is_file()


def test_bundled_sample_dataset_is_complete():
    folder = calibrate_module.DEFAULT_DATASET
    truth = read_truth(folder)
    assert len(truth) >= 30
    assert sum(len(lines) for lines in truth.values()) >= 80
    assert all((folder / name).is_file() for name in truth)
