from synthetic_autoregressive_proto.config import PrototypeConfig
from synthetic_autoregressive_proto.proto_types import DecodedSequence, PositionEvidence


def _evidence_support_map(evidences: list[PositionEvidence]) -> dict[tuple[int, str], float]:
    return {
        (ev.position, cand.token): cand.support
        for ev in evidences
        for cand in ev.candidates
    }


def _sequence_evidence_score(seq: DecodedSequence, evidence_support: dict[tuple[int, str], float]) -> float:
    support = 0.0
    for pos, tok in enumerate(seq.tokens):
        support += evidence_support.get((pos, tok), 0.0)
    return support


def _sequence_score(seq: DecodedSequence, evidence_support: dict[tuple[int, str], float],
                    config: PrototypeConfig) -> float:
    return (
            config.rerank_weight_support * _sequence_evidence_score(seq, evidence_support)
            + config.rerank_weight_coherence * seq.coherence_score
            + config.rerank_weight_source_coverage * seq.source_coverage_score
    )


def _batch_duplicate_penalty(batch: list[DecodedSequence]) -> float:
    penalty = 0.0
    for idx, seq in enumerate(batch):
        current = set(seq.tokens)
        for prev in batch[:idx]:
            other = set(prev.tokens)
            union = len(current | other) or 1
            penalty += len(current & other) / union
    return penalty


def rerank_candidate_batches(
        candidate_groups: list[list[DecodedSequence]],
        evidences: list[PositionEvidence],
        config: PrototypeConfig,
) -> list[DecodedSequence]:
    """Jointly choose one decoded candidate per latent slot.

    This is the synthetic analogue of scoring a complete reconstructed batch
    against one mixed evidence object. It keeps a small beam of whole-batch
    hypotheses and penalizes slot collapse/duplicate sequences globally.
    """
    evidence_support = _evidence_support_map(evidences)
    beam: list[tuple[float, list[DecodedSequence]]] = [(0.0, [])]
    beam_size = max(1, config.rerank_batch_beam_size)

    for group in candidate_groups:
        if not group:
            continue
        next_beam = []
        for _, partial in beam:
            for seq in group:
                batch = partial + [seq]
                score = sum(_sequence_score(item, evidence_support, config) for item in batch)
                score -= config.rerank_duplicate_penalty * _batch_duplicate_penalty(batch)
                next_beam.append((score, batch))
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:beam_size]

    if not beam:
        return []
    return sorted(beam[0][1], key=lambda seq: seq.slot_id)


def rerank_batch(
        decoded: list[DecodedSequence],
        evidences: list[PositionEvidence],
        config: PrototypeConfig,
) -> list[DecodedSequence]:
    evidence_support = _evidence_support_map(evidences)

    def score(seq: DecodedSequence) -> float:
        return _sequence_score(seq, evidence_support, config)

    ranked = sorted(decoded, key=score, reverse=True)
    selected: list[tuple[float, DecodedSequence]] = []
    for seq in ranked:
        dup_penalty = 0.0
        current = set(seq.tokens)
        for _, prev in selected:
            other = set(prev.tokens)
            union = len(current | other) or 1
            overlap = len(current & other) / union
            dup_penalty = max(dup_penalty, overlap)
        adjusted_seq = DecodedSequence(
            slot_id=seq.slot_id,
            tokens=seq.tokens,
            support_score=seq.support_score - config.rerank_duplicate_penalty * dup_penalty,
            coherence_score=seq.coherence_score,
            source_coverage_score=seq.source_coverage_score,
        )
        adjusted_score = score(adjusted_seq) - config.rerank_duplicate_penalty * dup_penalty
        selected.append((adjusted_score, adjusted_seq))
    return [seq for _, seq in sorted(selected, key=lambda item: item[0], reverse=True)]
