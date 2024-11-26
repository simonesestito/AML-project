import torch
from ..llama import LlamaInstruct


@torch.no_grad()
def tokenize_prompts_fixed_length(llama: LlamaInstruct, prompt: str | list[str] | tuple[str]) -> dict:
    if not isinstance(prompt, list) and not isinstance(prompt, tuple):
        assert isinstance(prompt, str), f"Prompt must be a string. Found: {type(prompt)}"
        prompt = [prompt]

    tokenized_strings, _ = llama.tokenize(prompt, pad_to_max_length=70)

    # Check that every string ends with the padding token = the string wouldn't have required a bigger max_length in the tokenizer
    padding_final_token_id = 128007
    assert torch.all(tokenized_strings.input_ids[:, -1] == padding_final_token_id), 'Every string should end with the padding token'

    return tokenized_strings


@torch.no_grad()
def convert_internals_from_layers_to_batch_items(collected_internals: list[torch.Tensor]) -> torch.Tensor:
      '''
      Post-process collected first hidden states (or attentions of the input)
      They are a list where i-th item is the output of i-th layer to ALL batch items
      We want it to be a list where j-th item is the stack of all hidden_layers (or attention maps) of that batch.
      '''
      # Transpose to make the batch the first dimension, instead of the layer index
      return torch.stack(collected_internals).transpose(0, 1)