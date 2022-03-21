import os, numpy as np, argparse
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import network

from gensim.models import KeyedVectors as Word2Vec
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

import io
from PIL import Image
import cv2
from torchvision import transforms
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--HMDBPath', default='/workspace/hmdb51_avi/',   type=str, help='Input the HMDB dataset path.')
parser.add_argument('--UCFPath', default='/workspace/UCF-101_avi/',   type=str, help='Input the UCF dataset path.')
parser.add_argument('--datasetName', default='HMDB',   type=str, help='UCF or HMDB.')
parser.add_argument('--clip_len',     default=16,   type=int, help='Number of frames of each sample clip.')
parser.add_argument('--n_clips',     default=1,   type=int, help='Number of clips per video, 1 or 25 in the paper.')
parser.add_argument('--size',         default=112,  type=int,   help='Image size in input.')
parser.add_argument('--weights',      default='AURL662_checkpoint.pth.tar', type=str, help='Weights to load from a previously run.')
parser.add_argument('--wordsmodel',    default='/workspace/word2vec/GoogleNews-vectors-negative300.bin', type=str, help='words model')
parser.add_argument('--nltkPath',      default='/workspace/word2vec/nltk_data', type=str, help='nltk_data path')
opt = parser.parse_args()

nltk.data.path.append(opt.nltkPath)

class ResizeImgSeq(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)
    def __call__(self, img_list):
        return [self.worker(img) for img in img_list]
class StackImgSeq(object):
    def __init__(self, roll=False):
        self.roll = roll
    def __call__(self, img_list):
        if self.roll:
            tmp = np.stack(img_list[::-1], axis = 0)
        else:
            tmp = np.stack(img_list, axis = 0)
        return torch.from_numpy(tmp).permute(0, 3, 1 ,2).contiguous().float().div(255)
class SeqNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.norm = torchvision.transforms.Normalize(mean=mean, std=std, inplace=True)
    def __call__(self, tensor):
        for t in tensor:
            self.norm(t)
        return tensor
def get_transform_ours(opt):
    resnet_trans = torchvision.transforms.Compose([
        ResizeImgSeq((opt.size,opt.size)),
        StackImgSeq(),
        SeqNormalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
    return resnet_trans

def load_word2vec(opt):
    wv_model = Word2Vec.load_word2vec_format(opt.wordsmodel, binary=True)
    wv_model.init_sims(replace=True)
    return wv_model
def verbs2basicform(words):
    ret = []
    for w in words:
        analysis = wn.synsets(w)
        if any([a.pos() == 'v' for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'v')
        ret.append(w)
    return ret
def one_class2embed_ucf(name, wv_model):
    change = {
        'CleanAndJerk': ['weight', 'lift'],
        'Skijet': ['Skyjet'],
        'HandStandPushups': ['handstand', 'pushups'],
        'HandstandPushups': ['handstand', 'pushups'],
        'PushUps': ['pushups'],
        'PullUps': ['pullups'],
        'WalkingWithDog': ['walk', 'dog'],
        'ThrowDiscus': ['throw', 'disc'],
        'TaiChi': ['taichi'],
        'CuttingInKitchen': ['cut', 'kitchen'],
        'YoYo': ['yoyo'],
    }
    if name in change:
        name_vec = change[name]
    else:
        upper_idx = np.where([x.isupper() for x in name])[0].tolist()
        upper_idx += [len(name)]
        name_vec = []
        for i in range(len(upper_idx)-1):
            name_vec.append(name[upper_idx[i]: upper_idx[i+1]])
        name_vec = [n.lower() for n in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec

def one_class2embed_hmdb(name, wv_model):
    change = {'claping': ['clapping']}
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec
def classes2embedding(dataset_name, class_name_inputs, wv_model):
    if dataset_name == 'UCF':
        one_class2embed = one_class2embed_ucf
    elif dataset_name == 'HMDB':
        one_class2embed = one_class2embed_hmdb
    embedding = [one_class2embed(class_name, wv_model)[0] for class_name in class_name_inputs]
    embedding = np.stack(embedding)
    return embedding.squeeze()

def get_hmdb(opt):
    folder = opt.HMDBPath
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        dir = os.path.join(str(folder), label)
        if not os.path.isdir(dir): continue
        for fname in sorted(os.listdir(dir)):
            if fname[-4:] != '.avi':
                continue
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label.replace('_', ' '))

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    #print(fnames, labels, classes)
    return fnames, labels, classes

def get_ucf(opt):
    folder = opt.UCFPath
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        for fname in os.listdir(os.path.join(str(folder), label)):
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label)

    classes = np.unique(labels)
    #print(fnames, labels, classes)
    return fnames, labels, classes
def load_clips_tsn(fname, clip_len=16, n_clips=1, is_validation=False):
    capture = cv2.VideoCapture(fname)
    success, frame = capture.read()
    imgpack = []
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return_frame = frame.copy()
        encoded_image = Image.fromarray(frame.astype('uint8'))
        imgByteArr = io.BytesIO()
        encoded_image.save(imgByteArr, format='JPEG')
        encoded_image = imgByteArr.getvalue()
        imgpack.append(encoded_image)
        success, frame = capture.read()
    frame_count = len(imgpack)
    total_frames = frame_count
    sampling_period = max(total_frames // n_clips, 1)
    n_snipets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(0, max(1, sampling_period - clip_len), n_snipets)
    else:
        starts = np.zeros(n_snipets).astype('int32')
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len) for of, s in zip(offsets, starts)])
    selection = selection%len(imgpack)
    sub_imgpack = np.array(imgpack)[selection]
    frames = []
    for img in sub_imgpack:
        stream = io.BytesIO(img)
        img_stream = Image.open(stream)
        frames.append(img_stream)
    while len(frames) < clip_len * n_clips:
       frames = frames + frames[:clip_len]
    return frames

if __name__ == '__main__':
    if opt.datasetName == 'HMDB':
       fnames, labels, classes = get_hmdb(opt)
    elif opt.datasetName == 'UCF':
       fnames, labels, classes = get_ucf(opt)
    else:
       print('Input HMDB or UCF.')
    
    img_model = network.get_network()
    words_model = network.Backbone_words()
    model = network.AURL(img_model, words_model)
    model  = model.cuda()
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['state_dict'])
    _ = model.eval()
    wv_model = load_word2vec(opt)
    test_class_embedding = classes2embedding(opt.datasetName, classes, wv_model)
    with torch.no_grad():
      class_embedding = model.encode_words(torch.from_numpy(test_class_embedding).cuda())
      class_embedding = class_embedding.cpu().detach().numpy()

    
    
    transform = get_transform_ours(opt)
    label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
    label_array = np.array([label2index[label] for label in labels], dtype=int)
    with torch.no_grad():
      cnt = 0
      rcnt = 0
      for i in range(len(fnames)):
        fpath = fnames[i]
        fclsname = labels[i]
        flabel = label_array[i]
        buffer = load_clips_tsn(fpath, opt.clip_len, opt.n_clips, is_validation=True)
        buffer = transform(buffer)
        buffer = buffer.reshape(opt.n_clips, opt.clip_len, 3, opt.size, opt.size).transpose(1, 2)
        buffer = torch.unsqueeze(buffer, 0)
        Visembed = model(buffer.cuda())
        Visembed = Visembed.cpu().detach().numpy()
        y_pred = cdist(Visembed, class_embedding, 'cosine')
        pred_label = y_pred[0].argmin(0)
        cnt = cnt + 1
        if pred_label == flabel:
           rcnt = rcnt+1
        print(opt.datasetName+' Top-1 acc: '+str(rcnt/cnt*100), 'pred: '+classes[pred_label], 'label: '+classes[flabel])
        
        
        
        

