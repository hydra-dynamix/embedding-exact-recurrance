import torch


def get_contextual_positions(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the original model position represented by each contextual row."""
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise ValueError(
            "attention_mask must have shape [1, model_position_count]"
        )

    return torch.where(attention_mask[0].bool())[0]


def get_token_state_vectors(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Return the raw contextual state for every actual token position.

    Shape:
        [token_count, representation_dimensions]

    Normalization is deliberately left to the detector layer so the model's
    native contextual states, including their magnitudes, remain available.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, model_position_count]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")

    with torch.no_grad():
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    final_layer_states = model_output.hidden_states[-1][0]
    contextual_positions = get_contextual_positions(attention_mask)
    raw_token_state_vectors = final_layer_states[contextual_positions]

    return raw_token_state_vectors.float()
