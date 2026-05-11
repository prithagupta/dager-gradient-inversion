from pathlib import Path

import json
from dataclasses import dataclass
from typing import Any

from synthetic_autoregressive_proto.proto_types import CandidateToken, PositionEvidence

Graph = dict[tuple[int, str], dict[tuple[int, str], float]]


@dataclass
class EvidenceRecord:
    input_index: int
    evidences: list[PositionEvidence]
    graph: Graph | None = None
    references: list[str] | None = None
    metadata: dict[str, Any] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_source_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    _require(isinstance(value, list), "candidate.source_ids must be a list of integers")
    source_ids = []
    for item in value:
        _require(isinstance(item, int), "candidate.source_ids must contain only integers")
        source_ids.append(item)
    return tuple(source_ids)


def _parse_candidate(raw: dict[str, Any], position: int) -> CandidateToken:
    _require(isinstance(raw, dict), "candidate entries must be objects")
    token = raw.get("token")
    support = raw.get("support", 1.0)
    _require(isinstance(token, str) and token, "candidate.token must be a non-empty string")
    _require(isinstance(support, (int, float)), "candidate.support must be numeric")
    return CandidateToken(
        token=token,
        position=position,
        support=float(support),
        source_ids=_as_source_ids(raw.get("source_ids")),
        is_noisy=bool(raw.get("is_noisy", False)),
    )


def _parse_positions(raw_positions: Any) -> list[PositionEvidence]:
    _require(isinstance(raw_positions, list), "input.positions must be a list")
    evidences = []
    seen_positions = set()
    for raw_pos in raw_positions:
        _require(isinstance(raw_pos, dict), "position entries must be objects")
        position = raw_pos.get("position")
        _require(isinstance(position, int) and position >= 0, "position.position must be a non-negative integer")
        _require(position not in seen_positions, f"duplicate position {position}")
        seen_positions.add(position)
        raw_candidates = raw_pos.get("candidates")
        _require(isinstance(raw_candidates, list) and raw_candidates, f"position {position} must contain candidates")
        candidates = [_parse_candidate(candidate, position) for candidate in raw_candidates]
        evidences.append(PositionEvidence(position=position, candidates=candidates))
    return sorted(evidences, key=lambda ev: ev.position)


def _parse_node(value: Any, field_name: str) -> tuple[int, str]:
    _require(isinstance(value, list) and len(value) == 2, f"{field_name} must be [position, token]")
    position, token = value
    _require(isinstance(position, int) and position >= 0, f"{field_name}[0] must be a non-negative integer")
    _require(isinstance(token, str) and token, f"{field_name}[1] must be a non-empty token")
    return position, token


def _parse_graph(raw_edges: Any) -> Graph | None:
    if raw_edges is None:
        return None
    _require(isinstance(raw_edges, list), "input.edges must be a list")
    graph: Graph = {}
    for raw_edge in raw_edges:
        _require(isinstance(raw_edge, dict), "edge entries must be objects")
        left = _parse_node(raw_edge.get("from"), "edge.from")
        right = _parse_node(raw_edge.get("to"), "edge.to")
        weight = raw_edge.get("weight", 1.0)
        _require(isinstance(weight, (int, float)), "edge.weight must be numeric")
        graph.setdefault(left, {})[right] = float(weight)
    return graph


def load_evidence_file(path: str | Path) -> dict[int, EvidenceRecord]:
    evidence_path = Path(path)
    with evidence_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    _require(isinstance(payload, dict), "evidence file root must be an object")
    raw_inputs = payload.get("inputs")
    _require(isinstance(raw_inputs, list) and raw_inputs, "evidence file must contain a non-empty inputs list")

    records: dict[int, EvidenceRecord] = {}
    for raw_input in raw_inputs:
        _require(isinstance(raw_input, dict), "input entries must be objects")
        input_index = raw_input.get("input_index")
        _require(isinstance(input_index, int) and input_index >= 0, "input.input_index must be a non-negative integer")
        _require(input_index not in records, f"duplicate input_index {input_index}")

        references = raw_input.get("references")
        if references is not None:
            _require(isinstance(references, list), "input.references must be a list of strings")
            _require(all(isinstance(ref, str) for ref in references), "input.references must be a list of strings")

        records[input_index] = EvidenceRecord(
            input_index=input_index,
            evidences=_parse_positions(raw_input.get("positions")),
            graph=_parse_graph(raw_input.get("edges")),
            references=references,
            metadata=raw_input.get("metadata") if isinstance(raw_input.get("metadata"), dict) else None,
        )
    return records
