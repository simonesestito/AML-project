import torch
import torch.nn as nn
import torch.nn.functional as F
from .hidden_states import LlamaHiddenStatesExtractor
from .tokenizer import get_tokenization_attention_masks

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

        # Apply softmax to weight matrix: sum over ALL dimensions is 1 
        weight_matrix = F.softmax(self.weight_matrix, dim=0).view((self.num_layers, self.num_tokens))
        
        return torch.einsum('blnd, ln -> bd', hidden_states, weight_matrix)


class AttentionAwareWeightedMeanReduction(nn.Module):
    def __init__(self, num_layers: int = 16, num_tokens: int = 70):
        super(AttentionAwareWeightedMeanReduction, self).__init__()

        self.num_layers = num_layers
        self.num_tokens = num_tokens
        
        # Define learnable weight matrix (num_layers x num_tokens)
        self.weight_matrix = nn.Parameter(torch.rand(num_layers * num_tokens))  # 16 x 70

    def forward(self, statements: tuple[str], extractor: LlamaHiddenStatesExtractor, modeldtype) -> torch.Tensor:
        """
        Apply weighted mean across layers and tokens
        """
        all_layers = set(range(16))
        hidden_states = extractor.extract_input_hidden_states_for_layers(prompt=statements, for_layers=all_layers).to(dtype=modeldtype).detach()

        # Also, apply attention mask to the hidden states, so that we can ignore padding tokens
        # => attention_mask: [BATCH_SIZE, SEQ_LEN]
        # We want to exclude padding tokens from the mean calculation
        # Weight matrix [layers x tokens] = [16 x 70]
        attn_mask = get_tokenization_attention_masks(extractor.llama, prompt=statements).detach()
        # versione scifu
        # weight_matrix = self.weight_matrix.view((self.num_layers, self.num_tokens)).unsqueeze(0).expand(attn_mask.shape[0], -1, -1) # [BATCH_SIZE, 16, 70]
        # expanded_mask = attn_mask.unsqueeze(1).expand(-1, self.num_layers, self.num_tokens) # [BATCH_SIZE, 16, 70]
        # attn_weights = torch.where(expanded_mask == 0, torch.tensor(float('-inf'), device=attn_mask.device), weight_matrix) # [BATCH_SIZE, 16, 70]
        #versione simo stramba(a detta di scifu)
        weight_matrix = self.weight_matrix.view((self.num_layers, self.num_tokens))[None, :, :]
        print(f"weight_matrix.shape post view and unsqueeze: {weight_matrix.shape}")
        attn_mask = attn_mask[:, None, :]
        print(f"attn_mask.shape post unsqueeze(1): {attn_mask.shape}")
        attn_weights = weight_matrix * attn_mask - 1000 * (1 - attn_mask) 
        print(f"attn_weights.shape post multiplication: {attn_weights.shape}")
        # Apply softmax to weight matrix: sum over ALL dimensions is 1
        weight_matrix = F.softmax(attn_weights.view(-1, self.num_layers*self.num_tokens), dim=1).view((-1,self.num_layers, self.num_tokens))
        
        return torch.einsum('blnd, bln -> bd', hidden_states, weight_matrix)
