import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import numpy as np
import random

BATCH_SIZE = 32
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


trainData = TensorDataset(torch.Tensor(train_data[2]), torch.LongTensor(train_data[0]))
# testData = TensorDataset(torch.Tensor(test_data[3]), torch.LongTensor(test_data[1]))
print('to tensor done')

trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False)
print('dataloader done')

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 1-2 conv layer
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 1 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            # 2-1 conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 2-2 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 2 Pooling lyaer
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(

            # 3-1 conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 3-2 conv layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 3 Pooling layer
            # nn.MaxPool2d(kernel_size=2, stride=2))
            nn.AvgPool2d(kernel_size=24, stride=24))
        

        self.layer6 = nn.Sequential(

            # 6 Fully connected layer
            nn.Linear(256, 256),
            # nn.Dropout(),
            nn.ReLU())


        self.layer7 = nn.Sequential(

            # 7 Fully connected layer
            nn.Linear(256*11, 1024),
            nn.Dropout(),
            nn.ReLU())

        self.layer8 = nn.Sequential(

            # 8 output layer
            nn.Linear(1024, 7),
            nn.Softmax())

    def forward(self, x):
        # x.shape ==(-1,10,96,96,3)
        out = []
        for k in range(11):
            #print(k)
            f = x[:,k,:,:,:]
            f = f.permute(0,3,1,2) 
            #（-1，3，96，96）
            f = self.layer1(f)  
            # (-1, 64, 48, 48)
            f = self.layer2(f)  
            # (-1, 128, 24, 24)
            f = self.layer3(f)  
            # (-1, 256, 1, 1)
            f = f.view(f.size(0),-1)
            # (-1, 256)
            f = self.layer6(f)
            # (-1, 256)
            out.append(f)
        out = torch.cat(out,axis=1) 
        #  (-1, 256*10)
        out = self.layer7(out)
        # (-1, 1024)
        out = self.layer8(out)
        # (-1, 7)
        return out
    
import torch.optim as optim


        


vgg16 = VGG16()

# checkpoint = torch.load('./RAFDB/vgg_dropout_epoch329.pkl') 
# vgg16.load_state_dict(checkpoint['net'], strict=False)

vgg16.cuda()
                               

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
    if epoch > 350:
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
        
#     for i in range(400):
#         index = np.random.randint(0, 12198, size=BATCH_SIZE)
#         # images = trainregion[index]
#         images = np.array(train_data[3]).reshape(-1, 10, 96, 96, 3)[index]
#         images = Variable(torch.Tensor(images)).cuda()
#         labels = np.array(train_data[1])[index]
#         labels = Variable(torch.LongTensor(labels)).cuda()
        
#         out = vgg16.forward(images)
#         # print(out.shape,labels.shape)
#         loss = cost(out, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         _, predicted = torch.max(out.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
        
    
#     loss_list.append(loss.data)
#     acc_list.append(correct / total)
    print('epoch: %d' % epoch, loss.data)
    print('Test Accuracy of the model on the train images: %d %%' % (100 * correct / total))
    if (epoch+1) % 10 == 0 :
        saved_dict = {
            'net': vgg16.state_dict(),
            'opt': optimizer.state_dict()
        }
        torch.save(saved_dict, './RAFDB/vgg_dropout_allepoch{}.pkl'.format(epoch))
        # torch.save(vgg16.state_dict(), './RAFDB/vgg_dropout_epoch{}.pkl'.format(epoch))
        
# Test the model
# vgg16.eval()
# correct = 0
# total = 0

# for images, labels in testLoader:
#     images = Variable(images).cuda()
#     outputs = vgg16(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()

# print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))

# Save the Trained Model
