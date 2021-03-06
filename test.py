import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
from IPython import embed
import torch.nn.functional as F

with open('./raf_test_96.pkl', 'rb') as f:
     test_data = pickle.load(f)
print('load test done')

# testData = TensorDataset((torch.Tensor(test_data[2]))[:, :10], torch.LongTensor(test_data[0]))
testData = TensorDataset(torch.Tensor(test_data[2]), torch.LongTensor(test_data[0]))
print('to tensor done')

testLoader = DataLoader(dataset=testData, batch_size=2, shuffle=False)
print('dataloader done')


class VGG16(nn.Module):  # vgg-11
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(33, 64, kernel_size=3, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7),
            nn.Softmax()
        )
        

    def forward(self, x):
        inp = []
        for k in range(11):
            f = x[:,k,:,:,:]
            f = f.permute(0,3,1,2) 
            inp.append(f)
        inp = torch.cat(inp, axis=1)
        out = self.features(inp)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



vgg16 = VGG16()
vgg16 = torch.nn.DataParallel(vgg16)

path = './RAFDB/vgg11allepoch189.pkl'
checkpoint = torch.load(path)

vgg16.load_state_dict(checkpoint['net'])
vgg16.cuda()

# Test the model
vgg16.eval()

correct = list(0. for i in range(7))
total = list(0. for i in range(7))
for i, (images, labels) in enumerate(testLoader):
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())

    output = vgg16(images)

    prediction = torch.argmax(output, 1)
    res = prediction == labels
    for label_idx in range(len(labels)):
        label_single = labels[label_idx]
        correct[label_single] += res[label_idx].item()
        total[label_single] += 1
acc_str = 'Accuracy: %f'%(sum(correct)/sum(total))
print(correct)
print(total)
for acc_idx in range(7):
    try:
        acc = correct[acc_idx]/total[acc_idx]
        print(acc)
    except:
        acc = 0
    finally:
        acc_str += '\tclassID:%d\tacc:%f\t'%(acc_idx+1, acc)
print(acc_str)


# correct = [0 for _ in range(7)]
# total = [0 for _ in range(7)]

# for images, labels in testLoader:
#     images = Variable(images).cuda()
#     outputs = vgg16(images)
#      _, predicted = torch.max(outputs.data, 1)
#     index = 
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()
# print(correct, total)
# print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))










# for i in range(9, 109, 10):
#     print(i)
#     path = './RAFDB/vgg11allepoch189' + str(i) +'.pkl'
#     checkpoint = torch.load(path)
#     # from collections import OrderedDict
#     # new_check = OrderedDict()
#     # for k, v in checkpoint.items():
#     #     if 'module' not in k:
#     #         k = 'module.'+k
#     #     else:
#     #         k = k.replace('features.module.', 'module.features.')
#     #     new_check[k]=v

#     vgg16.load_state_dict(checkpoint['net'])
#     vgg16.cuda()

#     # Test the model
#     vgg16.eval()
#     correct = 0
#     total = 0

#     for images, labels in testLoader:
#         images = Variable(images).cuda()
#         outputs = vgg16(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted.cpu() == labels).sum()
#     print(correct, total)
#     print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))