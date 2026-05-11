from synthetic_autoregressive_proto.proto_types import PositionEvidence


def _compatibility(left_token: str, right_token: str, left_sources: tuple[int, ...],
                   right_sources: tuple[int, ...]) -> float:
    shared_prefix = 0
    for a, b in zip(left_token, right_token):
        if a != b:
            break
        shared_prefix += 1
    prefix_score = shared_prefix / max(len(left_token), len(right_token), 1)
    length_score = 1.0 / (1.0 + abs(len(left_token) - len(right_token)))
    type_score = 1.0 if left_token[:1].isalpha() == right_token[:1].isalpha() else 0.0
    left_set = set(left_sources)
    right_set = set(right_sources)
    overlap = len(left_set & right_set)
    union = len(left_set | right_set) or 1
    source_overlap = overlap / union
    lexical_bonus = 0.2 if left_token == right_token else 0.0
    return (
            0.05
            + 0.15 * prefix_score
            + 0.15 * length_score
            + 0.10 * type_score
            + 0.55 * source_overlap
            + lexical_bonus
    )


def build_token_graph(evidences: list[PositionEvidence], edge_threshold: float) -> dict[
    tuple[int, str], dict[tuple[int, str], float]]:
    graph: dict[tuple[int, str], dict[tuple[int, str], float]] = {}

    for current, nxt in zip(evidences, evidences[1:]):
        for left in current.candidates:
            left_key = (left.position, left.token)
            graph.setdefault(left_key, {})
            for right in nxt.candidates:
                support_scale = 0.5 + 0.5 * min(left.support + right.support, 1.0)
                score = _compatibility(left.token, right.token, left.source_ids, right.source_ids) * support_scale
                if score >= edge_threshold:
                    graph[left_key][(right.position, right.token)] = score

    return graph
