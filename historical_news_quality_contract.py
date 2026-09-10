"""Single outcome-blind quality contract for historical news articles.

The collection worker, preview/freeze builder, coverage reporting, and model-view
builder must use this module rather than reimplementing Tier logic.  The contract
uses only publication/provenance/content/event-link metadata; it never reads a
return, direction label, prediction, correctness target, role, or fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


CONTRACT_VERSION = "HIST_NEWS_QUALITY_BASE_V1_20260831"

TIER_A = "A"
TIER_B_FULL = "B_FULL"
TIER_B_HEADLINE = "B_HEADLINE"
TIER_C = "C"

CAUSAL_TIERS = frozenset({TIER_A, TIER_B_FULL})
UNKNOWN_PUBLISHER_KEYS = frozenset({"", "unknown", "n/a", "na", "none", "null"})
PRECISE_PUBLICATION_VALUES = frozenset({"EXACT_SECOND", "EXACT_MINUTE"})
MIN_HEADLINE_CHARS = 8
MIN_BODY_CHARS = 80


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if text.casefold() in {"nan", "nat", "<na>"}:
        return ""
    return text


def normalize_publisher_key(value: Any) -> str:
    return clean_text(value).casefold()


def publisher_is_known(value: Any) -> bool:
    return normalize_publisher_key(value) not in UNKNOWN_PUBLISHER_KEYS


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _valid_time(value: Any) -> bool:
    if value is None or clean_text(value) == "":
        return False
    if isinstance(value, datetime):
        return True
    return hasattr(value, "to_pydatetime")


def _time_leq(left: Any, right: Any) -> bool:
    if not _valid_time(left) or not _valid_time(right):
        return False
    try:
        return bool(left <= right)
    except (TypeError, ValueError):
        return False


def _seconds_between(left: Any, right: Any) -> float | None:
    if not _valid_time(left) or not _valid_time(right):
        return None
    try:
        return float((left - right).total_seconds())
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class HistoricalArticleQuality:
    contract_version: str
    tier: str
    causal_eligible: bool
    headline_only_aux: bool
    first_seen_unobserved: bool
    publisher_known: bool
    precise_high_confidence_publication: bool
    event_match_high: bool
    headline_substantive: bool
    body_substantive: bool
    provenance_available: bool
    published_before_cutoff: bool
    first_seen_observed: bool
    first_seen_before_cutoff: bool
    current_crawl_as_historical_first_seen: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def classify_historical_article_quality(
    *,
    canonical_publisher: Any,
    published_at_precision: Any,
    published_at_confidence: Any,
    published_at: Any,
    decision_cutoff: Any,
    event_match_quality: Any,
    headline: Any,
    body: Any,
    first_seen_at: Any = None,
    first_seen_observed: Any = False,
    captured_at: Any = None,
    provenance_available: Any = False,
) -> HistoricalArticleQuality:
    """Classify one article under the conservative BASE V1 contract.

    Tier A requires genuine historical first-seen evidence.  Tier B_FULL admits
    precise published-time causal rows when first-seen is unavailable.  A precise
    causal headline with insufficient body is retained as B_HEADLINE auxiliary,
    never as a default P(correct) training row.
    """

    publisher_known = publisher_is_known(canonical_publisher)
    precision = clean_text(published_at_precision).upper() or "UNKNOWN"
    confidence = clean_text(published_at_confidence).upper() or "UNKNOWN"
    precise_high = precision in PRECISE_PUBLICATION_VALUES and confidence == "HIGH"
    event_match_high = clean_text(event_match_quality).upper() == "HIGH"
    headline_substantive = len(clean_text(headline)) >= MIN_HEADLINE_CHARS
    body_substantive = len(clean_text(body)) >= MIN_BODY_CHARS
    provenance = _truthy(provenance_available)
    published_before = _time_leq(published_at, decision_cutoff)

    observed = _truthy(first_seen_observed) and _valid_time(first_seen_at)
    first_seen_before = observed and _time_leq(first_seen_at, decision_cutoff)
    crawl_delta = _seconds_between(first_seen_at, captured_at)
    publication_lag = _seconds_between(captured_at, published_at)
    current_crawl_misused = bool(
        observed
        and crawl_delta is not None
        and abs(crawl_delta) <= 5.0
        and publication_lag is not None
        and publication_lag > 3600.0
    )

    reasons: list[str] = []
    if not publisher_known:
        reasons.append("PUBLISHER_UNKNOWN")
    if not precise_high:
        reasons.append("PUBLICATION_TIMESTAMP_NOT_PRECISE_HIGH_CONFIDENCE")
    if not event_match_high:
        reasons.append("EVENT_RELATIONSHIP_NOT_HIGH_CONFIDENCE")
    if not headline_substantive:
        reasons.append("HEADLINE_NOT_SUBSTANTIVE")
    if not body_substantive:
        reasons.append("BODY_NOT_SUBSTANTIVE")
    if not published_before:
        reasons.append("NOT_PROVEN_PUBLISHED_BY_EVENT_T2")
    if observed and not first_seen_before:
        reasons.append("FIRST_SEEN_AFTER_EVENT_T2")
    if current_crawl_misused:
        reasons.append("CURRENT_CRAWL_MISUSED_AS_HISTORICAL_FIRST_SEEN")

    common = publisher_known and precise_high and event_match_high and headline_substantive and published_before
    first_seen_allowed = not observed or (first_seen_before and not current_crawl_misused)
    if common and body_substantive and observed and first_seen_before and not current_crawl_misused and provenance:
        tier = TIER_A
        causal = True
        headline_only = False
    elif common and body_substantive and not observed:
        tier = TIER_B_FULL
        causal = True
        headline_only = False
        reasons.append("FIRST_SEEN_UNOBSERVED")
    elif common and not body_substantive and first_seen_allowed:
        tier = TIER_B_HEADLINE
        causal = False
        headline_only = True
        if not observed:
            reasons.append("FIRST_SEEN_UNOBSERVED")
        reasons.append("HEADLINE_ONLY_AUX")
    else:
        tier = TIER_C
        causal = False
        headline_only = False
        if not observed:
            reasons.append("FIRST_SEEN_UNOBSERVED")
        if observed and first_seen_before and not provenance:
            reasons.append("FIRST_SEEN_PROVENANCE_UNVERIFIED")
        reasons.append("RETROSPECTIVE_AUX_ONLY")

    return HistoricalArticleQuality(
        contract_version=CONTRACT_VERSION,
        tier=tier,
        causal_eligible=causal,
        headline_only_aux=headline_only,
        first_seen_unobserved=not observed,
        publisher_known=publisher_known,
        precise_high_confidence_publication=precise_high,
        event_match_high=event_match_high,
        headline_substantive=headline_substantive,
        body_substantive=body_substantive,
        provenance_available=provenance,
        published_before_cutoff=published_before,
        first_seen_observed=observed,
        first_seen_before_cutoff=first_seen_before,
        current_crawl_as_historical_first_seen=current_crawl_misused,
        reasons=tuple(sorted(set(reasons))),
    )
