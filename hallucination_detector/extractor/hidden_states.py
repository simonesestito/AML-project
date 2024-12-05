import torch
from ..llama import LlamaInstruct
from .tokenizer import tokenize_prompts_fixed_length, convert_internals_from_layers_to_batch_items
from .types import PromptType

class LlamaHiddenStatesExtractor:

    def __init__(self, llama: LlamaInstruct):
        """
        Initialize Llama model with hooks to capture hidden states in intermediate decoder layers
        """
        self.llama = llama.eval()


    @torch.no_grad()
    def extract_input_hidden_states_for_layer(self, prompt: PromptType, for_layer: int) -> torch.Tensor:
        """
        Given a batch of prompts, with length BATCH_SIZE,
        extract the hidden states for the requested layer.

        The output tensor will have shape [BATCH_SIZE, SEQ_LEN, TOKEN_DIM].
        """
        return self.extract_input_hidden_states_for_layers(prompt, {for_layer}).squeeze(1)


    @torch.no_grad()
    def extract_input_hidden_states_for_layers(self, prompt: PromptType, for_layers: set[int]) -> torch.Tensor:
        """
        Given a batch of prompts, with length BATCH_SIZE,
        extract the hidden states for the L requested layers.

        The output tensor will have shape [BATCH_SIZE, L, SEQ_LEN, TOKEN_DIM].
        SEQ_LEN is the length of the input sequence, which is the same for all prompts in the batch, fixed to 70.
        TOKEN_DIM is the dimension of the hidden states, which is the same for all layers in the model, fixed to 2048.
        """
        if isinstance(for_layers, list) or isinstance(for_layers, tuple):
            for_layers = set(for_layers)
        assert isinstance(for_layers, set), f"Expected for_layers to be a set. Found: {type(for_layers)}"

        max_layers = len(self.llama.iter_layers())
        assert all(0 <= layer < max_layers for layer in for_layers), f"Expected all layers to be in range [0, {max_layers}). Found: {for_layers}"

        hidden_states = []

        def _collect_hidden_states(layer_idx: int):
            def _hook(module, inputs, outputs):
                assert isinstance(outputs, tuple), f"Expected outputs to be a tuple. Found: {type(outputs)}"
                assert len(outputs) >= 1, f"Expected outputs to have 1+ elements. Found: {len(outputs)}"

                hidden_state = outputs[0]
                assert isinstance(hidden_state, torch.Tensor), f"Expected hidden_state to be a torch.Tensor. Found: {type(hidden_state)}"
                assert hidden_state.size(1) == 70 and hidden_state.size(2) == 2048, f"Expected hidden_state to have shape (?, 70, 2048). Found: {hidden_state.shape}"
                hidden_states.append(hidden_state)
            return _hook
        
        self.llama.unregister_all_hooks()
        for layer_idx, decoder_layer in enumerate(self.llama.iter_layers()):
            if layer_idx in for_layers:
                self.llama.register_hook(decoder_layer, _collect_hidden_states(layer_idx))

        inputs = tokenize_prompts_fixed_length(self.llama, prompt)
        _ = self.llama.generate(
            inputs,
            generate_args={
                "max_length": None,
                "max_new_tokens": 1,
                "num_return_sequences": 1,
                # We are collecting hidden_states in a more fine-grained way with hooks
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict_in_generate": False,
            }
        )
        self.llama.unregister_all_hooks()

        # Now, hidden_states are a list of tensors, each tensor representing the hidden_state for a layer we requested
        return convert_internals_from_layers_to_batch_items(hidden_states)
