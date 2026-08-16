import unittest
from types import SimpleNamespace

import torch

from src.contextual_state_vectors import get_contextual_state_vectors
from src.detector import (
    build_detector_bank,
    calibrate_detector_thresholds,
    represent_experience,
)
from src.token_state_vectors import get_token_state_vectors


class FakeModel(torch.nn.Module):
    def __init__(self, final_layer_states: torch.Tensor) -> None:
        super().__init__()
        self.final_layer_states = final_layer_states

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        if not output_hidden_states:
            raise AssertionError("hidden states must be requested")
        return SimpleNamespace(hidden_states=(self.final_layer_states,))


class ContextualStateTests(unittest.TestCase):
    def test_extraction_preserves_raw_magnitude_and_contextual_positions(self) -> None:
        states = torch.tensor([[[3.0, 4.0], [99.0, 99.0], [8.0, 15.0]]])
        model = FakeModel(states)
        input_ids = torch.tensor([[10, 0, 20]])
        attention_mask = torch.tensor([[1, 0, 1]])

        extracted = get_contextual_state_vectors(
            model,
            input_ids,
            attention_mask,
        )

        torch.testing.assert_close(
            extracted,
            torch.tensor([[3.0, 4.0], [8.0, 15.0]]),
        )
        torch.testing.assert_close(extracted.norm(dim=1), torch.tensor([5.0, 17.0]))
        self.assertIs(get_token_state_vectors, get_contextual_state_vectors)


class DetectorBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_references = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        self.bank = build_detector_bank(
            memory_state_sequences=[self.raw_references],
            source_episode_ids=["episode-a"],
            source_offset_sequences=[torch.tensor([2, 5])],
        )

    def test_bank_keeps_raw_references_and_source_provenance(self) -> None:
        torch.testing.assert_close(
            self.bank.raw_reference_vectors,
            self.raw_references,
        )
        self.assertEqual(self.bank.source_episode_ids, ["episode-a", "episode-a"])
        torch.testing.assert_close(self.bank.source_offsets, torch.tensor([2, 5]))
        self.assertFalse(hasattr(self.bank, "normalized_reference_vectors"))

    def test_calibration_and_representation_use_raw_dot_products(self) -> None:
        thresholds = calibrate_detector_thresholds(
            calibration_state_sequences=[torch.tensor([[3.0, 4.0], [1.0, 0.0]])],
            detector_bank=self.bank,
            quantile=0.5,
        )
        torch.testing.assert_close(thresholds, torch.tensor([25.0, 8.0]))

        model = FakeModel(
            torch.tensor([[[6.0, 8.0], [0.0, 10.0], [5.0, 0.0]]])
        )
        experience = represent_experience(
            model=model,
            input_ids=torch.tensor([[11, 12, 13]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
            detector_bank=self.bank,
            detector_thresholds=thresholds,
        )

        torch.testing.assert_close(
            experience.raw_contextual_state_vectors,
            torch.tensor([[6.0, 8.0], [0.0, 10.0], [5.0, 0.0]]),
        )
        torch.testing.assert_close(
            experience.detector_scores,
            experience.raw_contextual_state_vectors @ self.raw_references.T,
        )
        self.assertEqual(experience.detector_scores[0, 0].item(), 50.0)
        self.assertGreater(experience.detector_scores[0, 0].item(), 1.0)
        torch.testing.assert_close(
            self.bank.raw_reference_vectors,
            self.raw_references,
        )
        self.assertEqual(experience.detector_scores.shape, (3, 2))
        self.assertEqual(experience.detector_margins.shape, (3, 2))
        self.assertEqual(experience.coactivation_states.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
