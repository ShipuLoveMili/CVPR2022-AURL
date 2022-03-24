import os
import io
import argparse
import lmdb
import msgpack
import msgpack_numpy
from PIL import Image
import numpy as np
import random
import cv2
msgpack_numpy.patch()

def get_lmdb(filepath_csv, savelmdbpath, save_exitid_path):
    
    env = lmdb.open(savelmdbpath, readonly=False, map_size=1e12)
    txn = env.begin(write=True)
    fs = open(save_exitid_path,'w')
    with open(filepath_csv,'r') as f:
         cnt = 0
         for line in f:
             print(cnt)
             cnt = cnt+1
             words = line.strip('\n').split('\t')
             key = words[0]
             path = words[2]
             imglist = []
             capture = cv2.VideoCapture(path)
             success, frame = capture.read()
             while success:
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 return_frame = frame.copy()
                 #encode_success, encoded_image = cv2.imencode(".jpg", frame)
                 #print('test1:',return_frame)
                 encoded_image = Image.fromarray(frame.astype('uint8'))
                 #print('test2:',np.array(encoded_image))
                 #print(fdsafasd)
                 imgByteArr = io.BytesIO()
                 encoded_image.save(imgByteArr, format='JPEG')
                 encoded_image = imgByteArr.getvalue()
                 imglist.append(encoded_image)
                 success, frame = capture.read()
             if len(imglist)>0:
                 txn.put(key=key.encode(), value=msgpack.dumps(imglist))
                 fs.write(key+'\t'+savelmdbpath+'\n')
                 fs.flush()
             if cnt%100==0:
                print ('commit...')
                txn.commit()
                txn = env.begin(write=True)
             #if cnt==6:
                #break
    #print('test3',return_frame)
    print ('commit...')
    txn.commit()
    print ('commit end')
    env.close()
    fs.close()
    return key,return_frame

def test_read(lmdb_file,key,return_frame,fileindex,tstimgpath='/apdcephfs/share_1324356/shipu/action_lmdbfiles/testlmdb/'):
    env = lmdb.open(lmdb_file, lock=False, readonly=True)
    cnt=0
    with env.begin(write=False) as txn:
         val = txn.get(key.encode())
         oridata = msgpack.loads(val)
         #print(len(oridata))
         length = len(oridata)
         for i in range(length):
             ele = oridata[i]
             with open(tstimgpath+fileindex+str(cnt)+'.jpg','wb') as f:
                  f.write(ele)
                  #print('read success')
             cnt = cnt + 1
         print(cnt)
         return_frame=Image.fromarray(return_frame.astype('uint8'))
         return_frame.save(tstimgpath+'return_frame.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='getimglmdb')
    parser.add_argument("--fileindex", type=str, default="", help="setting")
    args = parser.parse_args()
    fileindex = args.fileindex
    filepath_csv = '/apdcephfs/share_1324356/shipu/action_lmdbfiles/processfiles/idfiles/splitid'+fileindex
    savelmdbpath = '/apdcephfs/share_1324356/shipu/action_lmdbfiles/lmdb/splitid_path'+fileindex
    save_exitid_path = '/apdcephfs/share_1324356/shipu/action_lmdbfiles/lmdb/existid_'+fileindex+'_.txt'
    key,returnframe = get_lmdb(filepath_csv, savelmdbpath,save_exitid_path)
    test_read(savelmdbpath,key,returnframe,fileindex)
