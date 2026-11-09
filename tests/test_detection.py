import json

import numpy as np
import pandas as pd

from src import config

TOL_DAYS = 1


def _planted():
    with open(config.PLANTED_JSON) as f:
        return json.load(f)


def _matches(cluster, p):
    return (
        cluster.kind == p["kind"]
        and abs((cluster.start - pd.Timestamp(p["start"])).days) <= TOL_DAYS
        and abs((cluster.end - pd.Timestamp(p["end"])).days) <= TOL_DAYS
    )


def test_every_planted_anomaly_is_recovered(results):
    for p in _planted():
        assert any(_matches(c, p) for c in results.clusters), f"missed planted anomaly {p['id']}"


def test_no_false_clusters_at_default_settings(results):
    planted = _planted()
    false = [c.label for c in results.clusters if not any(_matches(c, p) for p in planted)]
    assert not false, f"false clusters: {false}"
    assert len(results.clusters) == len(planted)


def test_comparison_stats_are_sane(results):
    for c in results.clusters:
        assert np.isfinite([c.xfold, c.pct_below, c.ci_low, c.ci_high]).all()
        assert 0 <= c.pct_below <= 100
        assert c.ci_low <= c.mean_adj <= c.ci_high
        assert len(c.ref_means) > 100
        if c.kind == "spend_cluster":
            assert c.xfold > 1
        else:
            assert c.xfold < 1


def test_reference_windows_never_overlap_a_cluster(results):
    d = results.daily
    for c in results.clusters:
        prefix = "out" if c.kind == "spend_cluster" else "in"
        excluded = d[f"{prefix}_in_cluster"].astype(float)
        clean_windows = (excluded.rolling(c.n_days).max() == 0).sum()
        assert len(c.ref_means) == clean_windows


def test_parameters_are_configurable(results):
    from src.analysis import DetectionConfig, detect_clusters

    _, strict = detect_clusters(results.daily, DetectionConfig(z_threshold=50))
    assert strict == []
    _, long_runs = detect_clusters(results.daily, DetectionConfig(min_run=4))
    assert all(c.n_days >= 4 for c in long_runs)
