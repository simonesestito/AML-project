import torch
import torch.nn as nn
import torch.nn.functional as F
from .hidden_states import LlamaHiddenStatesExtractor


class WeightedMeanReduction(nn.Module):
    def __init__(self, num_layers: int = 16):   # Weight only by layers, and always pick token at index 64, after Notebook 4 experiments
        super(WeightedMeanReduction, self).__init__()

        self.num_layers = num_layers
        
        # Define learnable weight matrix (only for layers)
        self.weight_matrix = nn.Parameter(torch.rand(num_layers,))  # 16 layers


    def forward(self, statements: tuple[str], extractor: LlamaHiddenStatesExtractor, modeldtype) -> torch.Tensor:
        """
        Apply weighted mean across layers and tokens
        """
        all_layers = set(range(self.num_layers))
        hidden_states = extractor.extract_input_hidden_states_for_layers(prompt=statements, for_layers=all_layers).to(dtype=modeldtype).detach()

        # Consider only token at index 64
        hidden_states = hidden_states[:, :, 64, :].squeeze(2)

        # Apply softmax to weight matrix
        weight_matrix = F.softmax(self.weight_matrix, dim=0)

        # Apply weighted mean reduction
        return torch.einsum('bld, l -> bd', hidden_states, weight_matrix)
