from dataclasses import dataclass
from .prompt import LlamaPrompt

@dataclass
class LlamaResponse:
    """
      Class to represent a response given by the Llama model.
    """
    prompt: LlamaPrompt
    response: str