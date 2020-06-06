import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self,in_features, out_features, bias=True): #(shape256, 256)
        super(GraphConvolution, self).__init__()
        # (A*X*W)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # inputs (b,10,256)
        # ajd (b,10,10)
        x = x.view(-1,256)
        support = torch.mm(x, self.weight) #(b*10,256)
        support = support.view(-1,10,support.size(1))
        output = torch.bmm(adj, support) #(b,10,10)*(b,10.256)  (b,10,256)
        if self.bias is not None:
            #print(output.shape,self.bias.shape)
            return output + self.bias
        else:
            return output
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(256, 256)
        self.gc2 = GraphConvolution(256, 256)
        self.fc0 = nn.Linear(4608, 256)
        self.fc1 = nn.Linear(256*10, 1024)
        self.fc2 = nn.Linear(1024, 7)  
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x, adj):# (-1,10,4608)
        out = []
        for k in range(10):
            f = x[:,k,:] #(-1,4608)
            f = self.dropout(f)
            f = self.fc0(f) #(-1,256)
            f = F.relu(f)
            f = f.reshape(-1,1,256)
            out.append(f)
        out = torch.cat(out, axis=1) #(-1,10,256)
        out = self.dropout(F.relu(self.gc1(out, adj)))
        # out = self.dropout(self.gc2(out, adj)) #(-1,10,256)
        out = out.view(out.size(0),-1) #(-1,2560)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(self.dropout(out))
        out = F.softmax(out, dim=1)
        return out
    
    
def get_batch_graph(inp):
    assert len(inp.shape) == 3
    x=inp
    norm1 = torch.sum(x**2,axis=2).reshape((x.shape[0],-1,1))
    norm2 = torch.sum(x**2,axis=2).reshape((x.shape[0],1,-1))
    dist = torch.bmm(x,x.permute(0,2,1))*(-2)+norm1+norm2
    sigma2 = torch.mean(dist,axis=2).reshape(-1,10,1)
    dist = torch.exp(-dist/sigma2)
    return dist


    
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import numpy as np
import random
import torch.optim as optim

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCH = 100

with open('./raf_train_96.pkl', 'rb') as f:
     train_data = pickle.load(f)
print('load train done')
# trainData = TensorDataset(torch.Tensor(train_data[2]), torch.LongTensor(train_data[0]))
# # testData = TensorDataset(torch.Tensor(test_data[2]), torch.LongTensor(test_data[0]))
# print('to tensor done')

# trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# # testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False)
# print('dataloader done')

regions = torch.Tensor(train_data[2])# (12198,11,96,96,3)
re = regions[:, :10]
print('1')
points = torch.Tensor(train_data[1])# (12198,10,2)
print('2')
labels = torch.LongTensor(train_data[0])#(12198)
print('3')

adjs = get_batch_graph(points)



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
        return out

    


vgg16 = VGG16()
vgg16.cuda()
vgg16 = torch.nn.DataParallel(vgg16)
# from collections import OrderedDict
# new_check = OrderedDict()
# for k, v in checkpoint.items():
#     if 'module' not in k:
#         k = 'module.'+k
#     else:
#         k = k.replace('features.module.', 'module.features.')
#     new_check[k]=v
checkpoint = torch.load('./RAFDB/vgg11newepoch119.pkl')
vgg16.load_state_dict(checkpoint['net'])
for k,v in vgg16.named_parameters():
    v.requires_grad=False #固定参数
vgg16.eval()


gcn =  GCN()
gcn.cuda()
gcn.train()
gcn = torch.nn.DataParallel(gcn)
index = [i for i in range(12198)]
random.shuffle(index)
# Loss and Optimizer
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(gcn.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)


for epoch in range(100):
    if epoch > 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
#     if epoch > 80:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-5
            
    correct = 0
    total = 0
    for i in range(1524):
        tmp = index[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        img = Variable(re[tmp]).cuda()
        outputs = vgg16(img)  #(-1, 4608*10)
        outputs = outputs.reshape(-1, 10, 4608)
        label = Variable(labels[tmp]).cuda()
        adj = adjs[tmp].cuda()
        out = gcn.forward(outputs,adj)
        loss = cost(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum()
                         
    print('epoch: %d' % epoch, loss.data)
    print('Test Accuracy of the model on the train images: %d %%' % (100 * correct / total))
    if (epoch+1) % 10 == 0 :
        saved_dict = {
            'net': gcn.state_dict(),
            'opt': optimizer.state_dict()
        }
        torch.save(saved_dict, './RAFDB/gcn1epoch{}.pkl'.format(epoch))