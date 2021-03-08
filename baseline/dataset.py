import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pdb
import preprocess_v3 as preprocess
import pathlib

#alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
alphabet = '2345678abcdefghklmnpqrstuvwxy'

def img_loader(img_path):
    im = Image.open(img_path)
    proc = preprocess.gen(im)
    return proc
    # img = Image.open(img_path)
    # frame = [None]*16
    # for f in range(16):
    #     img.seek(f)
    #     frame[f] = np.array(img.convert('RGB'), dtype=np.float32)
    #     frame[f] = np.mean(frame[f], axis=2, keepdims=True)
    # # pdb.set_trace()
    # return np.concatenate((frame[3],
    #                        (frame[7] - frame[3] + 255) / 2,
    #                        (frame[11] - frame[7] + 255) / 2,
    #                        (frame[15] - frame[11] + 255) / 2),
    #                       axis=2)

def make_dataset(data_path, alphabet, num_class, num_char):
    img_names = pathlib.Path(data_path).glob('**/*.gif')
    samples = []
    for p in img_names:
        img_name = p.name
        img_path = p.as_posix()
        target_str = img_name.split('=')[0].lower()
        if len(target_str) != num_char:
            print(target_str, img_path)
            assert False
        target = []
        for char in target_str:
            vec = [0] * num_class
            c = alphabet.find(char)
            if c == -1:
                raise ValueError('unexpected char: %c' % char)
            vec[c] = 1
            target += vec
        samples.append((img_path, target))
    return samples

class CaptchaData(Dataset):
    def __init__(self, data_path, num_class=len(alphabet), num_char=4, 
                 transform=None, target_transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transform = transform
        self.target_transform = target_transform
        self.alphabet = alphabet
        self.samples = make_dataset(self.data_path, self.alphabet, 
                                    self.num_class, self.num_char)
        self.cachedsamples = [None]*len(self.samples)
    
    def __len__(self):
        return len(self.samples)*self.num_char
    
    def __getitem__(self, index):
        # pdb.set_trace()
        img_path, target = self.samples[index//4]
        if self.cachedsamples[index//4] is None:
            self.cachedsamples[index//4] = img_loader(img_path)
        img = np.expand_dims(self.cachedsamples[index//4][index%4], axis=2)
        if self.transform:
            img = 255-img # bg=0
            img = self.transform(img)
            img = 1-img # ToTensor convert 0~255 to 0~1
        else:
            img = ToTensor()(img)
        target = target[len(alphabet)*(index%4):len(alphabet)*(1+index%4)]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)
