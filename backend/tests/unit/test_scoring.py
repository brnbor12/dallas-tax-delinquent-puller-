"""Unit tests for the motivated seller scoring engine."""

from datetime import date, timedelta

import pytest

from scoring.engine import (
    IndicatorSnapshot,
    calculate_score,
    HOT_THRESHOLD,
    WARM_THRESHOLD,
)


def make_indicator(
    indicator_type: str,
    status: str = "active",
    days_ago: int = 15,
    amount_cents: int | None = None,
) -> IndicatorSnapshot:
    filing_date = date.today() - timedelta(days=days_ago)
    return IndicatorSnapshot(
        indicator_type=indicator_type,
        status=status,
        filing_date=filing_date,
        amount_cents=amount_cents,
    )


class TestCalculateScore:
    def test_no_indicators_returns_zero(self):
        result = calculate_score([])
        assert result["total_score"] == 0
        assert result["tier"] == "cold"
        assert result["indicator_count"] == 0

    def test_resolved_indicators_ignored(self):
        indicators = [make_indicator("tax_delinquent", status="resolved")]
        result = calculate_score(indicators)
        assert result["total_score"] == 0
        assert result["tier"] == "cold"

    def test_single_hot_indicator_produces_score(self):
        indicators = [make_indicator("foreclosure", days_ago=5)]
        result = calculate_score(indicators)
        assert result["total_score"] > 0
        assert "foreclosure" in result["breakdown"]

    def test_score_is_hot_with_multiple_strong_signals(self):
        indicators = [
            make_indicator("foreclosure", days_ago=10),
            make_indicator("tax_delinquent", days_ago=20, amount_cents=5_000_00),
            make_indicator("probate", days_ago=5),
        ]
        result = calculate_score(indicators)
        assert result["total_score"] >= HOT_THRESHOLD
        assert result["tier"] == "hot"

    def test_score_never_exceeds_100(self):
        indicators = [make_indicator(t) for t in [
            "foreclosure", "pre_foreclosure", "tax_delinquent", "probate",
            "lien", "eviction", "code_violation", "vacant", "absentee_owner",
            "price_reduction", "expired_listing", "days_on_market",
        ]]
        result = calculate_score(indicators)
        assert result["total_score"] <= 100

    def test_recency_decay_reduces_old_indicators(self):
        fresh = [make_indicator("pre_foreclosure", days_ago=10)]
        old   = [make_indicator("pre_foreclosure", days_ago=400)]
        fresh_score = calculate_score(fresh)["total_score"]
        old_score   = calculate_score(old)["total_score"]
        assert fresh_score > old_score

    def test_large_tax_debt_boosts_score(self):
        small = [make_indicator("tax_delinquent", amount_cents=1_000_00)]
        large = [make_indicator("tax_delinquent", amount_cents=60_000_00)]
        small_score = calculate_score(small)["total_score"]
        large_score = calculate_score(large)["total_score"]
        assert large_score > small_score

    def test_duplicate_indicator_types_kept_at_highest(self):
        # Two pre_foreclosure indicators — should not double-count
        indicators = [
            make_indicator("pre_foreclosure", days_ago=5),
            make_indicator("pre_foreclosure", days_ago=100),
        ]
        result = calculate_score(indicators)
        single = calculate_score([make_indicator("pre_foreclosure", days_ago=5)])
        # Score with two should equal score with the fresher one (max is kept)
        assert abs(result["total_score"] - single["total_score"]) < 0.01

    def test_warm_tier_threshold(self):
        indicators = [make_indicator("code_violation", days_ago=45)]
        result = calculate_score(indicators)
        # code_violation alone at ~85% recency = ~12.75 weight -> score ~12.75 -> cold
        assert result["tier"] in {"cold", "warm"}

    def test_breakdown_keys_match_indicator_types(self):
        types = ["pre_foreclosure", "tax_delinquent", "vacant"]
        indicators = [make_indicator(t) for t in types]
        result = calculate_score(indicators)
        for t in types:
            assert t in result["breakdown"]
