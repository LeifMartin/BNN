import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import pandas as pd

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def test_ensemble(net,dtest,TEST_SAMPLES,TEST_BATCH_SIZE,BATCH_SIZE, CLASSES,DEVICE,shape,classification = True):
    net.eval()
    correct = 0
    TEST_SIZE = len(dtest)
    correct3 = 0
    cases3 = 0
    Numpy = 1
    if torch.is_tensor(dtest):
        Numpy = 0

    corrects = np.zeros(TEST_SAMPLES + 1, dtype=float)
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(dtest.shape[0] / BATCH_SIZE))):
            batch = (batch + 1)
            _x = dtest[old_batch: BATCH_SIZE * batch, shape[0]:shape[1]]
            _y = dtest[old_batch: BATCH_SIZE * batch, shape[2]:shape[3]]

            old_batch = BATCH_SIZE * batch


            if Numpy:
                data = Variable(torch.FloatTensor(_x)).cuda()
                target = Variable(torch.transpose(torch.LongTensor(_y), 0, 1).cuda())[0]

            else:
                data   = _x.to(DEVICE)
                target = _y.to(DEVICE)
                target = torch.transpose(target,0,1).long()[0]
            if (classification == True):
                outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
                for i in range(TEST_SAMPLES):
                    outputs[i] = net(data, sample=True)

                    if (i == 0):
                        mydata_means = sigmoid(outputs[i].detach().cpu().numpy())
                        for j in range(TEST_BATCH_SIZE):
                            mydata_means[j] /= np.sum(mydata_means[j])
                    else:
                        tmp = sigmoid(outputs[i].detach().cpu().numpy())
                        for j in range(TEST_BATCH_SIZE):
                            tmp[j] /= np.sum(tmp[j])
                        # print(sum(tmp[j]))
                        mydata_means = mydata_means + tmp

                mydata_means /= TEST_SAMPLES

                outputs[TEST_SAMPLES] = net(data, sample=False)
                output = outputs[0:TEST_SAMPLES].mean(0)
                preds = outputs.max(2, keepdim=True)[1]
                pred = output.max(1, keepdim=True)[1]  # index of max log-probability
                corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
                correct += pred.eq(target.view_as(pred)).sum().item()

            else:
                outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, 1).to(DEVICE)
                for i in range(TEST_SAMPLES):
                    outputs[i] = net(data, sample=True)

                    if (i == 0):
                        mydata_means = outputs[i].detach().cpu().numpy()
                        for j in range(TEST_BATCH_SIZE):
                            mydata_means[j] /= np.sum(mydata_means[j])
                    else:
                        tmp = outputs[i].detach().cpu().numpy()
                        for j in range(TEST_BATCH_SIZE):
                            tmp[j] /= np.sum(tmp[j])
                        # print(sum(tmp[j]))
                        mydata_means = mydata_means + tmp

                mydata_means /= TEST_SAMPLES

                outputs[TEST_SAMPLES] = net(data, sample=False)
                output = outputs[0:TEST_SAMPLES].mean(0)
                #preds = outputs[0].type(torch.float32)
                #pred = output[0].type(torch.float32)  # index of max log-probability
                
                criterion = nn.MSELoss()
                
                loss = torch.sqrt(criterion(output.to(DEVICE), _y.to(DEVICE)))
                
                print('\n', 'This is the RSS:',loss.item())
                
                return loss, outputs, output
                
                #print()
                
                #corrects += preds.sum(dim=1).squeeze().cpu().numpy()
                #correct += pred.sum().item()


            # print(mydata_means[1][1])
            for jj in range(TEST_BATCH_SIZE):
                if mydata_means[jj][pred.detach().cpu().numpy()[jj]] >= 0.95:
                    correct3 += pred[jj].eq(target.view_as(pred)[jj]).sum().item()
                    cases3 += 1

            if cases3 == 0:
                cases3 += 1

    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        else:
            print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
    print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
    corrects = np.append(corrects, correct)
    corrects = np.append(corrects, correct3 / cases3)
    corrects = np.append(corrects, cases3)

    return corrects