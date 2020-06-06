import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import numpy as np
import random
import torch.optim as optim

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCH = 100

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

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=3),
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
            nn.Linear(512 * 3 * 3 * 10, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7),
            nn.Softmax()
        )
        

    def forward(self, x):
        out = []
        for k in range(10):
            f = x[:,k,:,:,:]
            f = f.permute(0,3,1,2) 
            f = self.features(f)
            f = f.view(f.size(0), -1)
            out.append(f)
        out = torch.cat(out, axis=1)
        out = self.classifier(out)
        return out
    


vgg16 = VGG16()

# checkpoint = torch.load('./RAFDB/vgg_dropout_epoch329.pkl') 
# vgg16.load_state_dict(checkpoint['net'], strict=False)

vgg16.cuda()
vgg16 = torch.nn.DataParallel(vgg16)
                               

# Loss and Optimizer
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
# optimizer.load_state_dict(checkpoint['opt'])
# optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
# Train the model
vgg16.train()
# loss_list = []
# acc_list = []
for epoch in range(0, EPOCH):
    if epoch > 80:
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
        torch.save(saved_dict, './RAFDB/vgg112newepoch{}.pkl'.format(epoch))
        # torch.save(vgg16.state_dict(), './RAFDB/vgg_dropout_epoch{}.pkl'.format(epoch))
        
