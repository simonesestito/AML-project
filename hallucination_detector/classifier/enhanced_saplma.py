import torch.nn as nn
import torch.nn.init as init

class EnhancedSAPLMAClassifier(nn.Module):
    """
    Enhanced SAPLMA Classifier with added dropout, batch normalization
    """
    def __init__(self, input_size: int = 2048, dropout: float = 0.3):
        super(EnhancedSAPLMAClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),       
            nn.ReLU(),
            nn.Dropout(dropout),          
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),       
            nn.ReLU(),
            nn.Dropout(dropout),           
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),        
            nn.ReLU(),
            nn.Dropout(dropout),           
            
            nn.Linear(64, 1),
            nn.Sigmoid()               
        )
        
        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)

    # def _initialize_weights(self):
    #     """
    #     Custom weight initialization for the linear layers.
    #     """
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             init.kaiming_uniform_(m.weight, nonlinearity='relu')
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)
