from synthetic_autoregressive_proto.proto_types import PositionEvidence, SlotAssignment


def infer_slots(
    evidences: list[PositionEvidence],
    n_slots: int,
    slot_stride: int = 5,
    candidate_width: int = 3,
    source_bonus: float = 0.8,
) -> list[SlotAssignment]:
    slots = [SlotAssignment(slot_id=i, target_source_id=i) for i in range(n_slots)]

    for evidence in evidences:
        ranked = sorted(evidence.candidates, key=lambda c: (-c.support, c.token))
        if not ranked:
            continue
        for slot in slots:
            scored = []
            for idx, cand in enumerate(ranked):
                target_source = slot.target_source_id
                has_source = target_source is not None and target_source in cand.source_ids
                rotation_bias = ((slot.slot_id + evidence.position * slot_stride + idx) % max(len(ranked), 1)) / max(len(ranked), 1)
                score = cand.support + (source_bonus if has_source else 0.0) - 0.05 * rotation_bias
                scored.append((score, cand))
            scored.sort(key=lambda item: (-item[0], -item[1].support, item[1].token))
            slot.tokens_by_position.setdefault(evidence.position, []).extend(
                cand for _, cand in scored[: min(candidate_width, len(scored))]
            )

    return slots
