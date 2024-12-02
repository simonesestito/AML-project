import torch
import torch.nn as nn
import torch.nn.functional as F
from .hidden_states import LlamaHiddenStatesExtractor

class WeightedMeanReduction(nn.Module):
    def __init__(self, num_layers: int = 16, num_tokens: int = 70):
        super(WeightedMeanReduction, self).__init__()

        self.num_layers = num_layers
        self.num_tokens = num_tokens
        
        # Define learnable weight matrix (num_layers x num_tokens)
        self.weight_matrix = nn.Parameter(torch.rand(num_layers * num_tokens))  # 16 x 70

    def forward(self, statements: tuple[str], extractor: LlamaHiddenStatesExtractor, modeldtype) -> torch.Tensor:
        """
        Apply weighted mean across layers and tokens
        """
        all_layers = set(range(16))
        hidden_states = extractor.extract_input_hidden_states_for_layers(prompt=statements, for_layers=all_layers).to(dtype=modeldtype)

        # Apply softmax to weight matrix
        self.weight_matrix = F.softmax(self.weight_matrix, dim=0)\
                .view((self.num_layers, self.num_tokens))  # sum over ALL dimensions is 1
        
        return torch.einsum('blnd, ln -> bd', hidden_states, self.weight_matrix)
