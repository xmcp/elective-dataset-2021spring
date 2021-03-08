import torch
import torch.nn as nn
from torch.autograd import Variable
from model import CNN
from dataset import CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomRotation, RandomAffine
import pdb

import time
import os
from tqdm import tqdm

from dataset import alphabet

batch_size = 128
base_lr = 0.001
max_epoch = 30
model_path = './checkpoints/model_%d.pth'
restor = False

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

def calculat_acc(output, target):
    output, target = output.view(-1, len(alphabet)), target.view(-1, len(alphabet))
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    tot = 0 
    totsucc = 0
    for i, j in zip(target, output):
        tot+=1
        if torch.equal(i, j):
            totsucc+=1
    #print(sum(correct_list), len(correct_list))
    return totsucc/tot

def train():
    transforms_train = Compose([ToTensor(), RandomAffine(10, translate=(0.02, 0.05))])
    train_dataset = CaptchaData('./set-train', transform=transforms_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, 
                             shuffle=True, drop_last=True)
    # train_data_loader.dataset.set_use_cache(use_cache=True)
    test_data = CaptchaData('./set-test', transform=None)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, 
                                  num_workers=0, shuffle=True, drop_last=True)
    # test_data_loader.dataset.set_use_cache(use_cache=True)

    print('train set', len(train_dataset))
    print('test set', len(test_data))

    cnn = CNN()
    if torch.cuda.is_available():
        cnn.cuda()
    if restor:
        cnn.load_state_dict(torch.load(model_path))
#        freezing_layers = list(cnn.named_parameters())[:10]
#        for param in freezing_layers:
#            param[1].requires_grad = False
#            print('freezing layer:', param[0])
    
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=base_lr, weight_decay=True)
    criterion = nn.MultiLabelSoftMarginLoss()
    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=.002, verbose=True)
    
    for epoch in range(max_epoch):
        start_ = time.time()
        
        loss_history = []
        acc_history = []
        cnn.train()
        for img, target in (train_data_loader):
            # pdb.set_trace()
            img = Variable(img)
            target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = calculat_acc(output, target)
            acc_history.append(float(acc))
            loss_history.append(float(loss))

        print('train_loss: {:.4}|train_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
                ))
        
        loss_history = []
        acc_history = []
        cnn.eval()

        with torch.no_grad():
            for img, target in (test_data_loader):
                img = Variable(img)
                target = Variable(target)
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                output = cnn(img)
                loss = criterion(output, target)
                
                acc = calculat_acc(output, target)
                acc_history.append(float(acc))
                loss_history.append(float(loss))

            test_loss = torch.mean(torch.Tensor(loss_history))

        print('test_loss: {:.4}|test_acc: {:.4}'.format(
                test_loss,
                torch.mean(torch.Tensor(acc_history)),
                ))
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time()-start_))
        torch.save(cnn.state_dict(), model_path % epoch)

        sche.step(test_loss)

if __name__=="__main__":
    train()
    pass
