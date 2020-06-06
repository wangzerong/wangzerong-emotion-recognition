import cv2
import pickle
import os
import numpy as np


### rafdb train 12271 12198

# img, label, points, regions
data_jaffe = [[0 for _ in range(12198)],[0 for _ in range(12198)],[0 for _ in range(12198)]] 
rootdir = './RAFDB/train'
count = 0
index = [0,1,8,9,16,17,26,27,46,47]
f = open('./RAFDB/list_patition_label.txt')
labels = f.readlines()
for j in range(len(labels)): #
    if labels[j].split(' ')[0].split('_')[0] == 'train':        
        imgpath = os.path.join(rootdir, labels[j].split(' ')[0])
        # print(imgpath)
        img = cv2.imread(imgpath)
        # data_jaffe[0][count] = cv2.resize(img, (96,96))      
        data_jaffe[0][count] = int(labels[j].split(' ')[1]) - 1
        lmkpath = os.path.join(rootdir, labels[j].split(' ')[0]+'.detect_ret')
        if os.path.isfile(lmkpath):
            ff = open(lmkpath, 'rb')
            obj = pickle.load(ff)
            if obj == None or imgpath == './RAFDB/train/train_07576.jpg' or imgpath == './RAFDB/train/train_11670.jpg':
                # print(lmkpath)
                continue
            obj = obj['landmark']
            ff.close()
            points = []
            regions = []
            for k in index:
                pointx = int(obj['points'][k]['x'])
                pointy = int(obj['points'][k]['y'])
                points.append((pointx, pointy))
                img_ = cv2.resize(img[max(0, pointy-48):min(256, pointy+48),max(0, pointx-48):min(256, pointx+48)], (112,112))
                regions.append(img_)  
            regions.append(cv2.resize(img, (112,112)))  
            data_jaffe[1][count] = points
            data_jaffe[2][count] = regions
            count += 1
            print(count)

# # data_jaffe = np.array(data_jaffe)
with open('./raf_train_96_new.pkl', 'wb') as f:
     pickle.dump(data_jaffe, f)



