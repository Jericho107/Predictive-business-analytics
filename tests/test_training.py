from predictive_analytics.training import chronological_split, generate_dataset, multi_seed_validation, training_evidence


def test_chronological_split_has_no_time_overlap():
    train, validation, test = chronological_split(generate_dataset(42))
    assert max(row.day_index for row in train) < min(row.day_index for row in validation)
    assert max(row.day_index for row in validation) < min(row.day_index for row in test)


def test_real_training_beats_lag7_baseline_on_validation_and_final_test():
    result = training_evidence(42)
    assert result["accepted"] is True
    assert result["validation"]["improvement_pct"] > 20
    assert result["final_test"]["improvement_pct"] > 20
    assert result["top_coefficients"]


def test_multi_seed_acceptance_is_stable():
    results = multi_seed_validation()
    assert len(results) == 3
    assert all(result["accepted"] for result in results)
