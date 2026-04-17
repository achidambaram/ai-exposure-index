from ai_exposure_api.data_pipeline import _risk_band


def test_risk_band_thresholds():
    assert _risk_band(0.10) == "Green"
    assert _risk_band(0.50) == "Yellow"
    assert _risk_band(0.80) == "Red"
