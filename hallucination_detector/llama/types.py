from typing import TypeAlias
from .prompt import LlamaPrompt

SingleLlamaInstructInput: TypeAlias = str | LlamaPrompt
LlamaInstructInput: TypeAlias = SingleLlamaInstructInput | list[SingleLlamaInstructInput] | tuple[SingleLlamaInstructInput]