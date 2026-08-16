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
            detector_scores=torch.tensor([[4.0, 9.0, 1.0], [1.0, 2.0, 3.0]]),
            detector_activations=torch.tensor(
                [[0.7, 0.9, 0.4], [0.4, 0.3, 0.2]]
            ),
            detector_margins=torch.tensor([[0.1, 0.3, -0.2], [-0.2, -0.3, -0.4]]),
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

    def test_prints_complete_position_population_and_provenance(self) -> None:
        tokenizer = FakeTokenizer()
        output = io.StringIO()

        print_population_debug(
            experience=self.experience,
            tokenizer=tokenizer,
            detector_bank=self.bank,
            stream=output,
        )

        rendered = output.getvalue()
        self.assertIn(
            "position=0 token='first' raw_vector_norm=5.000000 "
            "firing_detector_count=2",
            rendered,
        )
        self.assertIn("detector_population:", rendered)
        self.assertIn(
            "detector_id=1 raw_score=9.000000 activation=0.900000 "
            "margin=0.300000 firing=True source_episode='episode-b' "
            "source_offset=4",
            rendered,
        )
        self.assertIn(
            "detector_id=0 raw_score=4.000000 activation=0.700000 "
            "margin=0.100000 firing=True source_episode='episode-a' "
            "source_offset=2",
            rendered,
        )
        self.assertIn(
            "detector_id=2 raw_score=1.000000 activation=0.400000 "
            "margin=-0.200000 firing=False source_episode='episode-c' "
            "source_offset=6",
            rendered,
        )
        self.assertIn(
            "position=2 token='second' raw_vector_norm=2.000000 "
            "firing_detector_count=0",
            rendered,
        )
        second_position = rendered.split("position=2", maxsplit=1)[1]
        self.assertEqual(second_position.count("detector_id="), 3)
        self.assertIn("firing=False", second_position)
        self.assertEqual(rendered.count("detector_id="), 6)
        self.assertEqual(tokenizer.decoded_token_ids, [10, 20])


if __name__ == "__main__":
    unittest.main()
