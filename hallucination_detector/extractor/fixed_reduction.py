import torch
import torch.nn as nn
from typing import Union, Literal, TypeAlias
from .hidden_states import LlamaHiddenStatesExtractor

ReductionType: TypeAlias = Union[int, Literal['mean', 'last']]

class HiddenStatesReduction(nn.Module):
    def __init__(self, hidden_states_layer_idx: int, reduction: TypeAlias = 'last'):
        super(HiddenStatesReduction, self).__init__()
        self.reduction = reduction
        self.hidden_states_layer_idx = hidden_states_layer_idx
        
    def forward(self, statements: tuple[str], extractor: LlamaHiddenStatesExtractor, model_dtype) -> torch.Tensor:
        """
        Apply reduction on hidden states: mean or last token
        """
        # Extract statements hidden states
        hidden_states = extractor.extract_input_hidden_states_for_layer(
            prompt=statements, for_layer=self.hidden_states_layer_idx).to(dtype=model_dtype)
        
        if self.reduction == 'mean':
            return torch.mean(hidden_states, dim=1)
        elif self.reduction == 'last':
            self.reduction = -1

        if not isinstance(self.reduction, int):
            raise ValueError(f'Unknown reduction type: {self.reduction}')
        
        return hidden_states[:, self.reduction, :]
