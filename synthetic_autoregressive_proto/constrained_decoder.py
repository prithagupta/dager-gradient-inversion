from synthetic_autoregressive_proto.proto_types import DecodedSequence, SlotAssignment


def _simple_coherence(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    transitions = 0.0
    for left, right in zip(tokens, tokens[1:]):
        transitions += 1.0 / (1.0 + abs(len(left) - len(right)))
    return transitions / max(len(tokens) - 1, 1)


def decode_slot(
    slot: SlotAssignment,
    graph: dict[tuple[int, str], dict[tuple[int, str], float]],
    max_sequence_length: int,
    beam_size: int = 1,
    graph_weight: float = 0.75,
    repetition_penalty: float = 0.25,
    common_token_penalty: float = 0.5,
    source_bonus: float = 0.9,
) -> DecodedSequence:
    candidates = decode_slot_candidates(
        slot,
        graph,
        max_sequence_length=max_sequence_length,
        beam_size=beam_size,
        graph_weight=graph_weight,
        repetition_penalty=repetition_penalty,
        common_token_penalty=common_token_penalty,
        source_bonus=source_bonus,
    )
    return candidates[0] if candidates else DecodedSequence(
        slot_id=slot.slot_id,
        tokens=[],
        support_score=0.0,
        coherence_score=0.0,
        source_coverage_score=0.0,
    )


def decode_slot_candidates(
    slot: SlotAssignment,
    graph: dict[tuple[int, str], dict[tuple[int, str], float]],
    max_sequence_length: int,
    beam_size: int = 4,
    graph_weight: float = 0.75,
    repetition_penalty: float = 0.25,
    common_token_penalty: float = 0.5,
    source_bonus: float = 0.9,
) -> list[DecodedSequence]:
    beams = [([], 0.0, 0.0, None)]
    for pos in sorted(slot.tokens_by_position):
        candidates = sorted(slot.tokens_by_position[pos], key=lambda c: (-c.support, c.token))
        if not candidates:
            continue
        next_beams = []
        for chosen, support_score, source_matches, prev_key in beams:
            for cand in candidates:
                score = support_score + cand.support
                if prev_key is not None:
                    score += graph_weight * graph.get(prev_key, {}).get((cand.position, cand.token), 0.0)
                if cand.token in chosen:
                    score -= repetition_penalty
                score -= common_token_penalty * cand.support
                next_source_matches = source_matches
                if slot.target_source_id is not None and slot.target_source_id in cand.source_ids:
                    score += source_bonus
                    next_source_matches += 1.0
                if cand.is_noisy:
                    score -= 0.1
                next_beams.append((chosen + [cand.token], score, next_source_matches, (cand.position, cand.token)))
        next_beams.sort(key=lambda item: (item[1], _simple_coherence(item[0])), reverse=True)
        beams = next_beams[:max(1, beam_size)]
        if beams and len(beams[0][0]) >= max_sequence_length:
            break

    decoded = []
    seen = set()
    for chosen, support_score, source_matches, _ in beams:
        key = tuple(chosen)
        if key in seen:
            continue
        seen.add(key)
        coherence = _simple_coherence(chosen)
        if chosen:
            coherence = 0.7 * coherence + 0.3 * (len(set(chosen)) / len(chosen))
        source_coverage = source_matches / max(len(chosen), 1)
        decoded.append(
            DecodedSequence(
                slot_id=slot.slot_id,
                tokens=chosen,
                support_score=support_score,
                coherence_score=coherence,
                source_coverage_score=source_coverage,
            )
        )
    return sorted(
        decoded,
        key=lambda seq: (seq.support_score, seq.coherence_score, seq.source_coverage_score),
        reverse=True,
    )
