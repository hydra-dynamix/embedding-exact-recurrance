import io
import unittest

import torch

from src.debugging import print_population_debug
from src.detector import DetectorBank, RepresentedExperience


class FakeTokenizer:
    def __init__(self) -> None:
        self.decoded_token_ids: list[int] = []

    def decode(
        self,
        token_ids: list[int],
        *,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        self.assert_cleanup_disabled(clean_up_tokenization_spaces)
        self.decoded_token_ids.extend(token_ids)
        return {10: "first", 20: "second"}[token_ids[0]]

    @staticmethod
    def assert_cleanup_disabled(clean_up_tokenization_spaces: bool) -> None:
        if clean_up_tokenization_spaces:
            raise AssertionError("debug token decoding must not rewrite whitespace")


class PopulationDebugTests(unittest.TestCase):
    def setUp(self) -> None:
        raw_contextual_states = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        self.experience = RepresentedExperience(
            input_ids=torch.tensor([[10, 99, 20]]),
            attention_mask=torch.tensor([[1, 0, 1]]),
            contextual_positions=torch.tensor([0, 2]),
            raw_contextual_state_vectors=raw_contextual_states,
            detector_scores=torch.tensor([[0.4, 0.9, 0.1], [0.1, 0.2, 0.3]]),
            detector_margins=torch.tensor([[0.2, 0.8, -0.1], [-0.1, -0.2, -0.3]]),
            coactivation_states=torch.tensor(
                [[True, True, False], [False, False, False]]
            ),
        )
        self.bank = DetectorBank(
            raw_reference_vectors=torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
            ),
            source_episode_ids=["episode-a", "episode-b", "episode-c"],
            source_offsets=torch.tensor([2, 4, 6]),
        )

    def test_prints_position_population_and_strongest_detector_provenance(self) -> None:
        tokenizer = FakeTokenizer()
        output = io.StringIO()

        print_population_debug(
            experience=self.experience,
            tokenizer=tokenizer,
            detector_bank=self.bank,
            strongest_detector_count=1,
            stream=output,
        )

        rendered = output.getvalue()
        self.assertIn(
            "position=0 token='first' raw_vector_norm=5.000000 "
            "firing_detector_count=2",
            rendered,
        )
        self.assertIn("strongest_firing_detectors:", rendered)
        self.assertIn(
            "detector_id=1 margin=0.800000 source_episode='episode-b' "
            "source_offset=4",
            rendered,
        )
        self.assertNotIn("detector_id=0", rendered)
        self.assertIn(
            "position=2 token='second' raw_vector_norm=2.000000 "
            "firing_detector_count=0",
            rendered,
        )
        self.assertIn("strongest_firing_detectors: none", rendered)
        self.assertEqual(tokenizer.decoded_token_ids, [10, 20])

    def test_rejects_nonpositive_display_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            print_population_debug(
                experience=self.experience,
                tokenizer=FakeTokenizer(),
                detector_bank=self.bank,
                strongest_detector_count=0,
            )


if __name__ == "__main__":
    unittest.main()
