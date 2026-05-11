from pathlib import Path

import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic_autoregressive_proto.batch_reranker import rerank_candidate_batches
from synthetic_autoregressive_proto.config import PrototypeConfig
from synthetic_autoregressive_proto.constrained_decoder import decode_slot_candidates
from synthetic_autoregressive_proto.evidence_extractor import SyntheticEvidenceExtractor
from synthetic_autoregressive_proto.slot_inference import infer_slots
from synthetic_autoregressive_proto.token_graph import build_token_graph

TOY_BATCH = [
    "this movie was charming and funny".split(),
    "the film was slow but sincere".split(),
    "a clever story with warm performances".split(),
    "this plot felt thin yet engaging".split(),
]


def main() -> None:
    config = PrototypeConfig()
    random.seed(config.random_seed)

    extractor = SyntheticEvidenceExtractor(rng_seed=config.random_seed)
    evidences = extractor.extract(
        TOY_BATCH,
        top_k_per_position=config.top_k_per_position,
        frequency_penalty=config.evidence_frequency_penalty,
    )
    graph = build_token_graph(evidences, edge_threshold=config.graph_edge_threshold)
    slots = infer_slots(evidences, n_slots=config.n_slots, slot_stride=config.slot_stride)
    candidate_groups = [
        decode_slot_candidates(
            slot,
            graph,
            max_sequence_length=config.max_sequence_length,
            beam_size=config.decoder_beam_size,
            graph_weight=config.decoder_graph_weight,
            repetition_penalty=config.decoder_repetition_penalty,
            common_token_penalty=config.decoder_common_token_penalty,
        )
        for slot in slots
    ]
    reranked = rerank_candidate_batches(candidate_groups, evidences, config)

    print("Synthetic autoregressive prototype")
    print("=" * 60)
    print(f"n_positions={len(evidences)}")
    print(f"graph_nodes={len(graph)}")
    print(f"n_slots={len(slots)}")
    print()
    print("Top evidence per position:")
    for ev in evidences:
        summary = ", ".join(f"{cand.token}:{cand.support:.2f}" for cand in ev.candidates[:3])
        print(f"  pos {ev.position}: {summary}")
    print()
    print("Decoded slots:")
    for seq in reranked:
        print(
            f"  slot={seq.slot_id} support={seq.support_score:.3f} "
            f"coherence={seq.coherence_score:.3f} text={seq.text}"
        )


if __name__ == "__main__":
    main()
