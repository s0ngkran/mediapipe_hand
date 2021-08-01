from torchvision import transforms
import torch
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

class HandLandmark(nn.Module):
    def __init__(self, unuse) -> None:
        super().__init__()
        self.layer1 = nn.Linear(21*3, 60)
        self.layer2 = nn.Linear(60, 40)
        self.layer3 = nn.Linear(40, 30)
        self.layer4 = nn.Linear(30, 25)
        self.act = torch.relu
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        x = self.act(x)
        x = self.layer4(x)
        x = torch.sigmoid(x)
        return x

def test_forword():
    model = HandLandmark()
    input_tensor = torch.rand(2, 21*3)
    output = model(input_tensor)
    print(output.shape)

if __name__ =='__main__':
    test_forword()
