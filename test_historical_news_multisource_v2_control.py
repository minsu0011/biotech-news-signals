from historical_news_multisource_v2_control import (
    ALLOWED_PRIORITY_FIELDS,
    FORBIDDEN_SELECTION_FIELDS,
    build_collection_policy,
    build_event_time_policy,
)


def test_collection_policy_is_outcome_blind_and_freezes_order() -> None:
    policy = build_collection_policy("a" * 64)
    assert policy["outcome_blind_collection"] is True
    assert policy["outcome_fields_read"] == []
    assert policy["source_freeze_before_label_join"] is True
    assert policy["role_split_before_label_join"] is True
    assert policy["base_v1_membership_mutable"] is False
    assert set(ALLOWED_PRIORITY_FIELDS).isdisjoint(FORBIDDEN_SELECTION_FIELDS)
    assert "future_return" in policy["forbidden_selection_fields"]


def test_event_time_policy_cannot_use_labels_or_rewrite_v1() -> None:
    policy = build_event_time_policy("b" * 64)
    assert policy["outcome_blind"] is True
    assert policy["outcome_fields_read"] == []
    assert policy["base_v1_event_time_mutable"] is False
    assert policy["label_or_return_use_for_resolution"] is False
    assert policy["event_time_freeze_before_price_label"] is True
