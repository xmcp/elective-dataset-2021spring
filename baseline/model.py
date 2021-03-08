import torch
import torch.nn as nn

from dataset import alphabet

# assume image size: 52x52
# assume input image pixels [0, 1]
class CNN(nn.Module):
    def __init__(self, num_class=len(alphabet), num_char=1):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=(2, 2), stride=(1, 1)), # after: 52x52
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, 5, padding=(2, 2), stride=(2, 2)), # after: 26x26
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, padding=(2, 2), stride=(1, 1)), # after: 26x26
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, padding=(1, 1), stride=(2, 2)), # after: 12x12
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, padding=(2, 2), stride=(1, 1)), # after: 26x26
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=(2, 2), stride=(2, 2)), # after: 6x6
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.fc = nn.Linear(32*6*6, self.num_class*self.num_char)

    def forward(self, x):
        x = torch.transpose(x, 2, 3)
        x = self.conv(x)
        x = x.view(-1, 32*6*6)
        x = self.fc(x)
        return x
