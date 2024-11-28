from .prompt import LlamaPrompt

type SingleLlamaInstructInput = str | LlamaPrompt
type LlamaInstructInput = SingleLlamaInstructInput | list[SingleLlamaInstructInput] | tuple[SingleLlamaInstructInput]