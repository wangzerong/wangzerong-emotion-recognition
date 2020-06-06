import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import random

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCH = 100

# transform = transforms.Compose([
#     transforms.RandomSizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
#                          std  = [ 0.229, 0.224, 0.225 ]),
#     ])


with open('./raf_train_112.pkl', 'rb') as f:
     train_data = pickle.load(f)
print('load train done')

# with open('./raf_test.pkl', 'rb') as f:
#      test_data = pickle.load(f)
# print('load test done')


trainData = TensorDataset((torch.Tensor(train_data[2]))[:, :10], torch.LongTensor(train_data[0]))
# trainData = TensorDataset(torch.Tensor(train_data[2]), torch.LongTensor(train_data[0]))
print('to tensor done')

trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False)
print('dataloader done')

class VGG16(nn.Module):

    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*2*2*10, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
            # nn.Softmax()
        )

    def forward(self, x):
        out = []
        for k in range(10):
            f = x[:,k,:,:,:]
            f = f.permute(0,3,1,2) 
            f = self.features(f)
            f = f.view(f.size(0), 256 * 2 * 2)
            out.append(f)
        out = torch.cat(out, axis=1)
        out = self.classifier(out)
        return out



        


vgg16 = VGG16()
vgg16.cuda()
vgg16 = torch.nn.DataParallel(vgg16)

# Loss and Optimizer
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
# Train the model
vgg16.train()
# loss_list = []
# acc_list = []
for epoch in range(EPOCH):
    if epoch > 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
#     if epoch > 260:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-5
            
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(trainLoader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        out = vgg16.forward(images)
        #print(out.shape,labels.shape)
        loss = cost(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
    print('epoch: %d' % epoch, loss.data)
    print('Test Accuracy of the model on the train images: %d %%' % (100 * correct / total))
    if (epoch+1) % 10 == 0 :
        saved_dict = {
            'net': vgg16.state_dict(),
            'opt': optimizer.state_dict()
        }
        torch.save(saved_dict, './RAFDB/alexnet112newepoch{}.pkl'.format(epoch))
        # torch.save(vgg16.state_dict(), './RAFDB/vgg_dropout_epoch{}.pkl'.format(epoch))
        
