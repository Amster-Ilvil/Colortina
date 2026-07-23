"""Top-2 character match decision and ambiguity gating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchDecision:
    character: object | None
    distance: float
    second_distance: float
    score: float
    second_score: float
    margin: float
    matched: bool
    lock_allowed: bool
    forced: bool = False


def decide_match(scored: list[tuple[float, object]], *,
                 match_threshold: float, min_match_score: float,
                 min_margin: float, semantic_confidence: float,
                 min_semantic_conf: float,
                 forced_character=None) -> MatchDecision:
    ranked = sorted(scored, key=lambda item: item[0])
    if forced_character is not None:
        first_distance = 0.0
        first_character = forced_character
        alternatives = [d for d, ch in ranked if ch is not forced_character]
        second_distance = alternatives[0] if alternatives else 1.0
        forced = True
    elif ranked:
        first_distance, first_character = ranked[0]
        second_distance = ranked[1][0] if len(ranked) > 1 else 1.0
        forced = False
    else:
        return MatchDecision(None, 1.0, 1.0, 0.0, 0.0, 0.0, False, False)

    score = float(np.clip(1.0 - first_distance, 0.0, 1.0))
    second_score = float(np.clip(1.0 - second_distance, 0.0, 1.0))
    margin = float(max(0.0, second_distance - first_distance))
    matched = bool(forced or first_distance <= match_threshold)
    lock_allowed = bool(forced or (
        matched and score >= min_match_score and margin >= min_margin
        and semantic_confidence >= min_semantic_conf))
    return MatchDecision(first_character if matched else None,
                         float(first_distance), float(second_distance),
                         score, second_score, margin, matched,
                         lock_allowed, forced)
