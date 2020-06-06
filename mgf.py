import mgfpy as mgf
mgf.MegFace.init()
import cv2
import sys
import os 
from meghair.utils import io
import pickle
MEGFACE_MODEL_PATH = '/opt/megface-v2/data'
device_opt = {
        'dev_type': mgf.DeviceType.MGF_GPU,
        'dev_id': 0, 'stream_id': 0}
path = '/data/jupyter/fyp/BiShe/CK+YuanTu/surprised/'
imgs = os.listdir(path)

for img in imgs:
    file_path = os.path.join(path, img)
    cv2_img = cv2.imread(file_path)
    try:
        img_ = mgf.Image.from_cv2_image(cv2_img)
    except:
        print(img)
        # continue
        # exit()
    detect_ctx = mgf.DetectorContext(os.path.join(MEGFACE_MODEL_PATH, 'detector.middle.v3.conf'),settings={'device': device_opt})
    detect_ret = detect_ctx.detect(img_)
    #from IPython import embed;embed()
    try:
        pickle.dump(detect_ret['items'][0],open(os.path.join(path,img+'.detect_ret'),'wb'))
    except:
        pickle.dump(None,open(os.path.join(path,img+'.detect_ret'),'wb'))
