from __future__ import annotations

from alabwebgazer.preprocessing.qc import compute_quadrant_image_qc


def _points(x0: float, y0: float) -> list[dict[str, float]]:
    return [
        {"t": 600.0, "x": x0, "y": y0},
        {"t": 900.0, "x": x0 + 5.0, "y": y0 + 3.0},
        {"t": 1200.0, "x": x0 - 4.0, "y": y0 - 2.0},
    ]


def test_compute_quadrant_image_qc_smoke() -> None:
    inner_w = 1000.0
    inner_h = 800.0
    trials = [
        {
            "stimulus": "<div style='top:25%; left:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": _points(100.0, 100.0),
        },
        {
            "stimulus": "<div style='top:25%; right:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": _points(900.0, 100.0),
        },
        {
            "stimulus": "<div style='bottom:25%; left:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": _points(100.0, 700.0),
        },
        {
            "stimulus": "<div style='bottom:25%; right:25%'>img</div>",
            "innerWidth": inner_w,
            "innerHeight": inner_h,
            "webgazer_data": _points(900.0, 700.0),
        },
    ]

    qc = compute_quadrant_image_qc(
        trials,
        q1_pattern="top:25%; left:25%",
        q2_pattern="top:25%; right:25%",
        q3_pattern="bottom:25%; left:25%",
        q4_pattern="bottom:25%; right:25%",
        median_kernel=5,
        x_prob_thres=0.6,
        y_prob_thres=0.6,
        random_state=0,
    )

    assert qc.n_image_trials == 4
    assert qc.confusion_raw is not None
    assert len(qc.confusion_raw) == 4
    assert qc.raw_accuracy is not None
    assert qc.raw_accuracy >= 0.5

    # Preprocessed confusion may be None only if sklearn is unavailable.
    if qc.confusion_preproc is not None:
        assert len(qc.confusion_preproc) == 4
        assert qc.preproc_accuracy is not None
