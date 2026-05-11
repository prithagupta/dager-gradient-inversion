import random

from synthetic_autoregressive_proto.proto_types import CandidateToken, PositionEvidence


class SyntheticEvidenceExtractor:
    """Builds toy per-position token evidence from synthetic sequences."""

    def __init__(self, rng_seed: int = 42):
        self.rng = random.Random(rng_seed)

    def extract(
            self,
            sequences: list[list[str]],
            top_k_per_position: int,
            frequency_penalty: float = 0.35,
    ) -> list[PositionEvidence]:
        max_len = max(len(seq) for seq in sequences)
        evidences: list[PositionEvidence] = []
        global_counts: dict[str, float] = {}

        for seq in sequences:
            for tok in seq:
                global_counts[tok] = global_counts.get(tok, 0.0) + 1.0

        for pos in range(max_len):
            counts: dict[str, float] = {}
            source_map: dict[str, set[int]] = {}
            noisy_map: dict[str, bool] = {}
            for source_id, seq in enumerate(sequences):
                if pos < len(seq):
                    tok = seq[pos]
                    counts[tok] = counts.get(tok, 0.0) + 1.0
                    source_map.setdefault(tok, set()).add(source_id)
                    noisy_map.setdefault(tok, False)

                    # Add light synthetic ambiguity.
                    if len(tok) > 3:
                        noisy = tok[:-1]
                        counts[noisy] = counts.get(noisy, 0.0) + 0.15
                        source_map.setdefault(noisy, set()).add(source_id)
                        noisy_map[noisy] = True

            adjusted_counts = {}
            for token, score in counts.items():
                penalty = 1.0 + frequency_penalty * max(global_counts.get(token, 1.0) - 1.0, 0.0)
                noisy_penalty = 1.5 if noisy_map.get(token, False) else 1.0
                adjusted_counts[token] = score / (penalty * noisy_penalty)

            total = sum(adjusted_counts.values()) or 1.0
            ranked = sorted(adjusted_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k_per_position]
            candidates = [
                CandidateToken(
                    token=token,
                    position=pos,
                    support=score / total,
                    source_ids=tuple(sorted(source_map.get(token, ()))),
                    is_noisy=noisy_map.get(token, False),
                )
                for token, score in ranked
            ]
            evidences.append(PositionEvidence(position=pos, candidates=candidates))

        return evidences
