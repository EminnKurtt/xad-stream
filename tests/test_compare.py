import pandas as pd
import pytest
from test_baseline import _synthetic

from xad.compare import compare_models, flag_all_reference, summarise


def _two_experiments():
    a, b = _synthetic(seed=0), _synthetic(seed=1)
    a["experiment"], b["experiment"] = "toy/0", "toy/1"
    return pd.concat([a, b])


def test_compare_models_rows_and_summary():
    results = compare_models(_two_experiments())
    assert len(results) == 4
    assert set(results["model"]) == {"IF", "EIF"}
    assert (results["n_train"] == 400).all()
    assert (results["roc_auc"] > 0.95).all()

    means, eif_wins = summarise(results)
    assert list(means.index) == ["IF", "EIF"]
    assert list(means.columns) == ["precision", "recall", "f1", "roc_auc"]
    assert 0 <= eif_wins <= 2


def test_flag_all_reference():
    # rows 400..699 are evaluated, 150 of them (450..599) are anomalous
    ref = flag_all_reference(_two_experiments())
    assert ref["precision"] == pytest.approx(0.5)
    assert ref["recall"] == 1.0
    assert ref["f1"] == pytest.approx(2 / 3)
