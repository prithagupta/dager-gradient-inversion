from dataclasses import dataclass


@dataclass
class PrototypeConfig:
    n_slots: int = 4
    top_k_per_position: int = 4
    graph_edge_threshold: float = 0.20
    assignment_temperature: float = 0.8
    slot_stride: int = 5
    slot_candidate_width: int = 3
    slot_source_bonus: float = 0.8
    decoder_beam_size: int = 3
    max_sequence_length: int = 12
    decoder_graph_weight: float = 0.75
    decoder_repetition_penalty: float = 0.25
    decoder_common_token_penalty: float = 0.5
    decoder_source_bonus: float = 0.9
    evidence_frequency_penalty: float = 0.35
    rerank_weight_support: float = 1.0
    rerank_weight_coherence: float = 0.5
    rerank_weight_source_coverage: float = 0.75
    rerank_duplicate_penalty: float = 0.8
    rerank_batch_beam_size: int = 8
    random_seed: int = 42
