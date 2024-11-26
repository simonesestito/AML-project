import torch.nn as nn

class OriginalSAPLMAClassifier(nn.Module):
    """
      SAPLMA Classifier as defined in https://arxiv.org/pdf/2304.13734
    """
    def __init__(self, input_size:int=2048):
        super(OriginalSAPLMAClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)
