import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..llama import LlamaInstruct
from ..extractor import LlamaHiddenStatesExtractor


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
        self.reduction.train()

        # Xavier init all weights for all Linear
        print('[LightningSAPLMA] Initializing all weights for all Linear layers')
        for layer in [self.reduction, self.saplma_classifier]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, statements: tuple[str]):
        # We need to reduce the hidden states to a single tensor dimension for all the tokens
        model_dtype = next(self.saplma_classifier.parameters()).dtype
        reduced_hidden_states = self.reduction(statements, self.hidden_states_extractor, model_dtype)
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
