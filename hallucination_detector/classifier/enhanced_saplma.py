import torch
import torch.nn as nn


class EnhancedSAPLMAClassifier(nn.Module):
    """
    Enhanced SAPLMA Classifier with added dropout, batch normalization
    """

    DEFAULT_HIDDEN_STATE_SIZES = [256, 128, 64]

    def __init__(self, input_size: int = 2048, dropout: float = 0.3, norm: str = 'batch', hidden_sizes: list[int] = None):
        super(EnhancedSAPLMAClassifier, self).__init__()

        # Construct the architecture dynamically, as requested by the parameters
        layers = []
        for prev_hidden_size, curr_hidden_size in zip([input_size] + hidden_sizes[:-1], hidden_sizes, strict=True):
            layers.extend([
                nn.Linear(prev_hidden_size, curr_hidden_size),
                nn.BatchNorm1d(curr_hidden_size) if norm == 'batch' else nn.LayerNorm(curr_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # Add the final layer
        layers.extend([
            nn.Linear(hidden_sizes[-1], 1),
            nn.Sigmoid()
        ])

        self.classifier = nn.Sequential(*layers)

        # Apply weight initialization
        self._initialize_weights()


    def forward(self, x):
        return self.classifier(x)


    def _initialize_weights(self):
        """
        Custom weight initialization for the linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
