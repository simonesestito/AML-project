import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompt import LlamaPrompt
from .response import LlamaResponse
from typing import Iterator

class LlamaInstruct:
    """
    Class to wrap the Llama model methods for ease of usage
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", model_args: dict = None, tokenizer_args: dict = None, pad_token: str = None):

        self.model_name = model_name
        self.model_args = model_args if model_args is not None else dict()
        self.tokenizer_args = tokenizer_args if tokenizer_args is not None else dict()

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", **self.model_args)
        self.model.eval()
        self.device = self.model.device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', **self.tokenizer_args)
        self.pad_token = self.tokenizer.eos_token if pad_token is None else pad_token
        self.tokenizer.pad_token = self.pad_token

        self.assistant_header = self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", return_tensors="pt").to(self.device)

        self.registered_hooks = []

    def eval(self) -> 'LlamaInstruct':
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self

    # to tokenize input prompts
    def tokenize(self, prompts: str | LlamaPrompt | list[str | LlamaPrompt], pad_to_max_length: int = 70) -> tuple[dict, list[LlamaPrompt]]:

        # Make prompts a list anyway
        if not isinstance(prompts, list):
            prompts = [ prompts ]

        # Convert all prompts to LlamaPrompt
        prompts = [ prompt if isinstance(prompt, LlamaPrompt) else LlamaPrompt(prompt) for prompt in prompts ]

        # tokenizer output will be a dictionary of pytorch tensors with keys "input_ids" (numerical ids of tokens)
        # and "attention_mask" (1 for actual input tokens and 0 for padding tokens)
        inputs = self.tokenizer(
            [ str(prompt) for prompt in prompts ],
            truncation=True,
            return_tensors="pt",
            padding='max_length',
            max_length=pad_to_max_length,
        ).to(self.device)

        return inputs, prompts

    # to make Llama generate responses
    def generate(self, inputs: dict, generate_args: dict = None): #-> Iterator[LlamaResponse] ? right now it does not return that

        generate_args = generate_args if generate_args is not None else dict()
        default_args = {
            "max_length": 100,
            "num_return_sequences": 1,
            "temperature": 0.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Overwrite default_args with generate_args
        default_args.update(generate_args)

        # returns (batch_size, sequence_length) tensors with token ids of the generated response, including input tokens
        return self.model.generate(
            **inputs,
            **default_args,
        )

    # to extract Llama answers as decoded text in a LlamaResponse object starting from encoded input
    def extract_responses(self, input_ids: torch.Tensor, outputs: torch.Tensor, prompts: list[LlamaPrompt]) -> Iterator[LlamaResponse]:

        for input, output, prompt in zip(input_ids, outputs, prompts):
            # Remove the prompt from the output generated
            output = output[len(input):]

            # Remove another assistant_header, if present
            if torch.equal(output[:len(self.assistant_header)], self.assistant_header):
                output = output[len(self.assistant_header):]

            generated = self.tokenizer.decode(output, skip_special_tokens=True).strip()

            yield LlamaResponse(prompt, generated)

    # to get textual Llama responses starting from textual prompts
    def run(self, prompts: str | LlamaPrompt | list[str | LlamaPrompt], verbose: bool = False) -> Iterator[LlamaResponse]:

        # Optional logging function
        def _print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        inputs, prompts = self.tokenize(prompts)

        _print('Tokenized inputs:', inputs.input_ids.shape)
        _print('Last tokens:', inputs.input_ids[:, -1])

        outputs = self.generate(inputs)
        _print('Generated outputs:', outputs.shape)

        return self.extract_responses(inputs.input_ids, outputs, prompts)

    # an hook is a piece of customized code to be run in the forward or backward pass of a model
    # (useful for debugging)
    def register_hook(self, module, hook_fn):
        '''
        Register a hook, in such a way that we have a very easy way to remove the hook later.

        Example usage:

        llama.unregister_all_hooks()
        for module_name, module in llama.model.named_modules():
            if something():
                llama.register_hook(module, hook_fn)
        '''
        handle = module.register_forward_hook(hook_fn)
        self.registered_hooks.append(handle)

    def unregister_all_hooks(self):
        '''
        Remove all of our registered hooks.
        '''
        for handle in self.registered_hooks:
            handle.remove()

    def _get_model_num_heads(self) -> int:
        return self.model.config.num_attention_heads

    def _get_model_hidden_layers(self) -> int:
        return self.model.config.num_hidden_layers
    
    def iter_layers(self) -> nn.ModuleList:
        return next(iter(self.model.children())).layers