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
EPOCH = 400

# transform = transforms.Compose([
#     transforms.RandomSizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
#                          std  = [ 0.229, 0.224, 0.225 ]),
#     ])



# with open('./ck+.pkl', 'rb') as f:
#      data = pickle.load(f)
# print('load done')     

# index = [i for i in range(927)]
# random.shuffle(index)
# trainindex = index[:835]
# testindex = index[835:]

# # dic = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'sad':5, 'surprise':6}
# dic = {'angry':0, 'disgusted':1, 'fearful':2, 'happy':3, 'sadness':4, 'surprised':5}
# for i in range(len(data[1])):
#     data[1][i] = dic[data[1][i]]
# testlabel = np.array(data[1])[testindex]
# trainlabel = np.array(data[1])[trainindex]
# testdata = np.array(data[3])[testindex]
# traindata = np.array(data[3])[trainindex]

# trainData = TensorDataset(torch.Tensor(traindata), torch.LongTensor(trainlabel))
# testData = TensorDataset(torch.Tensor(testdata), torch.LongTensor(testlabel))
# print('to tensor done')

# trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=False)
# testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False)
# print('dataloader done')

with open('./raf_train.pkl', 'rb') as f:
     train_data = pickle.load(f)
print('load train done')

# with open('./raf_test.pkl', 'rb') as f:
#      test_data = pickle.load(f)
# print('load test done')


trainData = TensorDataset((torch.Tensor(train_data[2]))[:, :10], torch.LongTensor(train_data[0]))
# testData = TensorDataset(torch.Tensor(test_data[3]), torch.LongTensor(test_data[1]))
print('to tensor done')

trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False)
print('dataloader done')

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv0 = nn.Conv2d(30, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 128, 1)
        self.conv8 = nn.Conv2d(128, 256, 1)
        self.BN1 = nn.BatchNorm2d(64)
        self.BN2 = nn.BatchNorm2d(128)
        self.BN3 = nn.BatchNorm2d(256)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*12*12, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 7)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # x.shape ==(-1,10,96,96,3)
        inp = []
        for k in range(10):
            f = x[:,k,:,:,:]
            f = f.permute(0,3,1,2) #(-1,3,96,96)
            inp.append(f)
        inp = torch.cat(inp, axis=1) #(-1,30,96,96)
 
        #（-1，30，96，96）
        f1 = F.relu(self.BN1(self.conv0(inp))) 
        #（-1，64，96，96）
        f2 = F.relu(self.BN1(self.conv1(f1)))
        f3 = self.maxpool(F.relu(f1 + self.BN1(self.conv2(f2))))
        residual1 = self.BN2(self.conv7(f3)) # (-1, 128, 48, 48)
        # (-1, 64, 48, 48)
        f4 = F.relu(self.BN2(self.conv3(f3))) 
        f5 = self.maxpool(F.relu(residual1 + self.BN2(self.conv4(f4))))
        residual2 = self.BN3(self.conv8(f5)) # (-1, 256, 24, 24)    
        # (-1, 128, 24, 24)
        f6 = F.relu(self.BN3(self.conv5(f5))) 
        f7 = F.relu(residual2 + self.BN3(self.conv6(f6)))
        f8 = self.maxpool(f7)  
        # (-1, 256, 12, 12)
        f8 = f8.view(f8.size(0),-1)
        # (-1, 256*12*12)
        f9 = F.relu(self.fc1(self.dropout(f8)))
        # (-1, 1024)
        f10 = F.relu(self.fc2(self.dropout(f9)))
        # (-1, 1024)
        out = F.softmax(self.fc3(f10), dim=1)
        # (-1, 7)
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
    if epoch > 10:
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
        
    
#     loss_list.append(loss.data)
#     acc_list.append(correct / total)
    print('epoch: %d' % epoch, loss.data)
    print('Test Accuracy of the model on the train images: %d %%' % (100 * correct / total))
    if (epoch+1) % 10 == 0 :
        saved_dict = {
            'net': vgg16.state_dict(),
            'opt': optimizer.state_dict()
        }
        torch.save(saved_dict, './RAFDB/resnetnewepoch{}.pkl'.format(epoch))
        # torch.save(vgg16.state_dict(), './RAFDB/vgg_dropout_epoch{}.pkl'.format(epoch))
        
