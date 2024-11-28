import torch
from ..llama import LlamaInstruct
from .tokenizer import tokenize_prompts_fixed_length, convert_internals_from_layers_to_batch_items
from .types import PromptType

class LlamaAttentionExtractor:

    def __init__(self, llama: LlamaInstruct):
        """
        Initialize Llama model with hooks to capture attention maps
        """
        llama_attention = llama.model_args.get("attn_implementation", None)
        assert llama_attention == "eager", f"Expected model_args to have 'attn_implementation' = 'eager'. Found: {llama_attention}"
        self.llama = llama.eval()


    @torch.no_grad()
    def extract_input_attention_maps_for_layer(self, prompt: PromptType, for_layer: int) -> torch.Tensor:
        """
        Given a batch of prompts, with length BATCH_SIZE,
        extract the attention maps for the requested layer.

        The output tensor will have shape [BATCH_SIZE, SEQ_LEN, SEQ_LEN].
        """
        return self.extract_input_attention_maps_for_layers(prompt, {for_layer}).squeeze(1)


    @torch.no_grad()
    def extract_input_attention_maps_for_layers(self, prompt: PromptType, for_layers: list[int]) -> torch.Tensor:
        """
        Given a batch of prompts, with length BATCH_SIZE,
        extract the attention maps for the L requested layers.

        The output tensor will have shape [BATCH_SIZE, L, SEQ_LEN, SEQ_LEN].
        SEQ_LEN is the length of the input sequence, which is the same for all prompts in the batch, fixed to 70.
        """
        if isinstance(for_layers, set) or isinstance(for_layers, tuple):
            for_layers = list(for_layers)
        assert isinstance(for_layers, list), f"Expected for_layers to be a list. Found: {type(for_layers)}"

        max_layers = len(self.llama.iter_layers())
        assert all(0 <= layer < max_layers for layer in for_layers), f"Expected all layers to be in range [0, {max_layers}). Found: {for_layers}"

        inputs = tokenize_prompts_fixed_length(self.llama, prompt)
        outputs = self.llama.generate(
            inputs,
            generate_args={
                "max_length": None,
                "max_new_tokens": 1,
                "num_return_sequences": 1,
                # We are collecting attentions
                "output_attentions": True,
                "output_hidden_states": False,
                "return_dict_in_generate": True,
            }
        )

        # We are interested in the forward pass with only the input sequence, and no output yet.
        # This is the first attention maps, of every layer
        # Since we are requesting "max_new_tokens = 1",
        # the only forward pass that occurs is the one having the whole input (and nothing more) to process.
        collected_input_attentions = outputs.attentions[0]  # tuple[layers_num = 16] having Tensor[batch_size, heads = 32, input_tokens = 70, input_tokens = 70]

        # Take only the attention maps for the requested layers
        collected_input_attentions = [collected_input_attentions[layer] for layer in for_layers]

        # Now, attention_maps are a list of tensors, each tensor representing the attention map for a layer we requested
        return convert_internals_from_layers_to_batch_items(collected_input_attentions)
