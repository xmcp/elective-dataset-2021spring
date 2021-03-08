import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
import os
from model import CNN
import preprocess_v3 as preprocess

from dataset import alphabet

model = CNN()
model.load_state_dict(torch.load('./model_120.pth',
                                 map_location=torch.device('cpu')))
model.eval()

def process(path):
    data = preprocess.gen(Image.open(path))
    data = torch.stack([ToTensor()(np.expand_dims(c, axis=2)) for c in data], dim=0)
    pred = model(data).view(4, len(alphabet))
    return ''.join(alphabet[i] for i in torch.argmax(pred, dim=1))

if __name__ == '__main__':
    print(process('test-xsag.gif'))
