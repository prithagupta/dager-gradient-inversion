from dataclasses import dataclass, field


@dataclass
class CandidateToken:
    token: str
    position: int
    support: float
    source_ids: tuple[int, ...] = ()
    is_noisy: bool = False


@dataclass
class PositionEvidence:
    position: int
    candidates: list[CandidateToken] = field(default_factory=list)


@dataclass
class SlotAssignment:
    slot_id: int
    target_source_id: int | None = None
    tokens_by_position: dict[int, list[CandidateToken]] = field(default_factory=dict)


@dataclass
class DecodedSequence:
    slot_id: int
    tokens: list[str]
    support_score: float
    coherence_score: float
    source_coverage_score: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(self.tokens).strip()
