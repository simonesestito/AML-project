import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..llama import LlamaInstruct
from ..extractor import LlamaHiddenStatesExtractor, TokenReductionType


class LightningHiddenStateSAPLMA(pl.LightningModule):
    '''
    Lightning module that uses the hidden states of a Llama model to classify statements.
    '''
    def __init__(
        self,
        llama: LlamaInstruct,
        saplma_classifier: nn.Module,
        reduction: nn.Module,
        lr: float = 1e-5,
    ):
        super().__init__()
        llama.eval()
        self.hidden_states_extractor = LlamaHiddenStatesExtractor(llama)
        self.saplma_classifier = saplma_classifier
        self.reduction = reduction
        self.lr = lr
        self.save_hyperparameters('lr')

    def on_fit_start(self):
        self.saplma_classifier.train()

        # Xavier init all weights for all Linear
        print('[LightningSAPLMA] Initializing all weights for all Linear layers')
        for module in self.saplma_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, statements: tuple[str]):

        # We need to reduce the hidden states to a single tensor dimension for all the tokens
        model_dtype = next(self.saplma_classifier.parameters()).dtype
        reduced_hidden_states = self.reduction(statements, model_dtype)
        assert len(reduced_hidden_states.shape) == 2, \
            f'Expected reduced_hidden_states dimensions to be 2. Found: {reduced_hidden_states.shape}, with reduction: {self.reduction}'

        # Classify
        return self.saplma_classifier(reduced_hidden_states)

    def do_step(self, batch, prefix_str: str):
        statements, labels, _ = batch
        assert isinstance(statements, tuple), f'Expected statements to be a tuple. Found: {type(statements)}'
        assert isinstance(labels, torch.Tensor), f'Expected labels to be a tensor. Found: {type(labels)}'
        assert len(labels.shape) == 1, f'Expected labels to be a 1D tensor. Found: {labels.shape}'
        assert labels.size(0) == len(statements), \
            f'Expected labels to have the same size as statements. Found: {labels.size(0)} != {len(statements)}'

        preds = self.forward(statements).squeeze(1)
        labels = labels.to(dtype=preds.dtype)

        assert torch.all(torch.logical_or(labels == 0, labels == 1)), \
            f'Expected labels to be 0 or 1. Found: {labels}'
        assert labels.shape == preds.shape, \
            f'Expected labels and preds to have the same shape. Found: {labels.shape} != {preds.shape}'
        assert labels.dtype == preds.dtype, \
            f'Expected labels and preds to have the same dtype. Found: {labels.dtype} != {preds.dtype}'

        loss = F.binary_cross_entropy(preds, labels)
        self.log(f'{prefix_str}/loss', loss, prog_bar=True)

        acc = torch.mean((preds > 0.5).to(dtype=labels.dtype) == labels, dtype=torch.float32)
        self.log(f'{prefix_str}/acc', acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.do_step(batch, 'test')

    def configure_optimizers(self):
        assert self.hparams.batch_size is not None, 'batch_size must be set in model.hparams'
        return torch.optim.AdamW(self.parameters(), lr=self.lr * self.hparams.batch_size)

class HiddenStatesReduction(nn.Module):
    def __init__(self, hidden_states_layer_idx: int, reduction: str = 'last'):
        super(HiddenStatesReduction, self).__init__()
        self.reduction = reduction
        self.hidden_states_layer_idx = hidden_states_layer_idx
        
    def forward(self, statements: tuple[str], model_dtype) -> torch.Tensor:
        """
        Apply reduction on hidden states: mean or last token
        """
        # Extract statements hidden states
        hidden_states = self.hidden_states_extractor.extract_input_hidden_states_for_layer(
            prompt=statements, for_layer=self.hidden_states_layer_idx).to(dtype=model_dtype)
        match self.reduction:
            case 'mean':
                return torch.mean(hidden_states, dim=1)  # Mean across tokens
            case 'last':
                return hidden_states[:, -1, :]  # Last token hidden state
            case _:
                raise ValueError(f'Unknown reduction type: {self.reduction}')
                

class WeightedMeanReduction(nn.Module):
    def __init__(self, num_layers: int = 16, num_tokens: int = 70):
        super(WeightedMeanReduction, self).__init__()
        
        # Define learnable weight matrix (num_layers x num_tokens)
        self.weight_matrix = nn.Parameter(torch.randn(num_layers, num_tokens) * 0.01)

    def forward(self, statements: tuple[str], modeldtype) -> torch.Tensor:
        """
        Apply weighted mean across layers and tokens
        """
        
        hidden_states = self.hidden_states_extractor.extract_input_hidden_states_for_layers(prompt=statements, for_layers=set(x for x in range(16))).to(dtype=modeldtype)
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
