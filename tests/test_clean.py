import pandas as pd

from src import config


def test_cleaning_recovers_ground_truth(results):
    truth = pd.read_csv(config.TRUTH_CSV)
    truth["date"] = pd.to_datetime(truth["date"])
    truth = truth.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(results.transactions, truth, check_dtype=False)


def test_raw_file_is_actually_messy():
    raw = pd.read_csv(config.RAW_CSV, dtype=str, keep_default_na=False)
    assert raw["transaction_id"].duplicated().any()
    assert raw["amount"].str.contains(r"\$").any()
    assert raw["date"].str.contains("/").any()
    assert (raw["category"] == "").any()


def test_log_counts_add_up(results):
    log = results.log.to_frame().set_index("step")["rows_affected"]
    assert log["Load"] - log["Remove duplicates"] == log["Result"]
    assert log["Result"] == len(results.transactions)
