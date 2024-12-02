import torch
import torch.nn as nn
from .hidden_states import LlamaHiddenStatesExtractor

class WeightedMeanReduction(nn.Module):
    def __init__(self, num_layers: int = 16, num_tokens: int = 70):
        super(WeightedMeanReduction, self).__init__()
        
        # Define learnable weight matrix (num_layers x num_tokens)
        self.weight_matrix = nn.Parameter(torch.randn(num_layers, num_tokens) * 0.01)

    def forward(self, statements: tuple[str], extractor: LlamaHiddenStatesExtractor, modeldtype) -> torch.Tensor:
        """
        Apply weighted mean across layers and tokens
        """
        all_layers = set(range(16))
        hidden_states = extractor.extract_input_hidden_states_for_layers(prompt=statements, for_layers=all_layers).to(dtype=modeldtype)
        # Expand weight matrix for broadcasting across batch dimension and hidden dimension
        weight_matrix_expanded = self.weight_matrix.unsqueeze(0).unsqueeze(-1)  # Shape: (1, num_layers, num_tokens, 1)

        # Apply weights to the hidden states: multiply element-wise
        weighted_hidden_states = hidden_states * weight_matrix_expanded  # Shape: (batch_size, num_layers, num_tokens, hidden_dim)

        # Compute weighted sum of hidden states across layers and tokens
        weighted_sum = weighted_hidden_states.sum(dim=(1, 2))  # Sum over layers and tokens, resulting in (batch_size, hidden_dim)
        
        # Normalize by the sum of the weights (this gives a weighted mean)
        weight_sum = weight_matrix_expanded.sum(dim=(1, 2))  # Sum of weights over layers and tokens, resulting in (1, 1, 1, 1)
        
        weighted_mean = weighted_sum / weight_sum  # Shape: (batch_size, hidden_dim)
        
        return weighted_mean