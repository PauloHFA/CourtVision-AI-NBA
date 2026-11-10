from __future__ import annotations

from dataclasses import asdict, dataclass

from courtvision.ingestion.normalize import NormalizedGame


@dataclass
class QualityReport:
    games: int
    duplicate_source_ids: int
    orphan_team_ids: int
    final_missing_score: int
    same_team_both_sides: int
    invalid_dates: int
    quarantined: int
    inserted: int = 0
    updated: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.duplicate_source_ids == 0
            and self.orphan_team_ids == 0
            and self.final_missing_score == 0
            and self.same_team_both_sides == 0
            and self.invalid_dates == 0
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def build_quality_report(
    games: list[NormalizedGame],
    known_provider_team_ids: set[int],
    *,
    inserted: int = 0,
    updated: int = 0,
) -> QualityReport:
    seen: set[int] = set()
    duplicates = 0
    orphans = 0
    missing = 0
    same_team = 0
    invalid = 0
    quarantined = 0
    for game in games:
        if game.source_game_id in seen:
            duplicates += 1
        seen.add(game.source_game_id)
        if game.quarantine_reason:
            quarantined += 1
        if game.quarantine_reason == "same_team":
            same_team += 1
        if game.quarantine_reason == "final_missing_score":
            missing += 1
        if game.quarantine_reason == "invalid_date":
            invalid += 1
        if game.home_provider_team_id not in known_provider_team_ids or game.visitor_provider_team_id not in known_provider_team_ids:
            orphans += 1
    return QualityReport(
        games=len(games),
        duplicate_source_ids=duplicates,
        orphan_team_ids=orphans,
        final_missing_score=missing,
        same_team_both_sides=same_team,
        invalid_dates=invalid,
        quarantined=quarantined,
        inserted=inserted,
        updated=updated,
    )
