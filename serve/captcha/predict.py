import torch
import numpy as np
import os
from torchvision.transforms import ToTensor
from .model import CNN
from . import preprocess_v3 as preprocess

from .dataset import alphabet

MODEL_PATH = os.path.join(os.path.split(__file__)[0], 'model.pth')

model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def process(im):
    with torch.no_grad():
        data = preprocess.gen(im)
        data = torch.stack([ToTensor()(np.expand_dims(c, axis=2)) for c in data], dim=0)
        pred = model(data).view(4, len(alphabet))
        return ''.join(alphabet[i] for i in torch.argmax(pred, dim=1))