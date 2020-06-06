import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle

BATCH_SIZE =32
LEARNING_RATE = 0.01
EPOCH = 300

# transform = transforms.Compose([
#     transforms.RandomSizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
#                          std  = [ 0.229, 0.224, 0.225 ]),
#     ])



# with open('./ck+.pkl', 'rb') as f:
#      data = pickle.load(f)
        
        # trainlabel = data[1][:820] 
        # testlabel = data[1][820:]
        # # dic = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'sad':5, 'surprise':6}
        # dic = {'angry':0, 'disgusted':1, 'fearful':2, 'happy':3, 'sadness':4, 'surprised':5}
        # for i in range(len(testlabel)):
        #     testlabel[i] = dic[testlabel[i]]
        # for i in range(len(trainlabel)):
        #     trainlabel[i] = dic[trainlabel[i]] 
            
            # trainregion = data[3][:820]
            # testregion = data[3][820:]
with open('./raf_train.pkl', 'rb') as f:
    train_data = pickle.load(f)
    print('load train done')
    from IPython import embed;embed()
    # with open('./raf_test.pkl', 'rb') as f:
    #      test_data = pickle.load(f)
    # print('load test done')
    # trainData = TensorDataset(torch.Tensor(train_data[3]), torch.LongTensor(train_data[1]))
    # patch_data = torch.Tensor(train_data[3])
    # label_data = torch.LongTensor(train_data[1])
    # testData = TensorDataset(torch.Tensor(test_data[3]), torch.LongTensor(test_data[1]))
    print('to tensor done')
    trainLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    # testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False)
    print('dataloader done')
