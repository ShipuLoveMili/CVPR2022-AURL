import os, numpy as np, argparse, time, multiprocessing
import random
from tqdm import tqdm

import torch
import torch.nn as nn
#from tensorboardX import SummaryWriter

import network
import dataset
from auxiliary.transforms import batch2gif

from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

from colorama import Fore, Style
import horovod.torch as hvd
from horovod.torch.sync_batch_norm import SyncBatchNorm
# Initialize Horovod
hvd.init()
def get_optimizer(name, model, lr, momentum, weight_decay):

    predictor_prefix = ('module.prediction', 'prediction')
    parameters = [{
            'name': 'base',
            'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
            'lr': lr
        },{
            'name': 'prediction',
            'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
            'lr': lr
        }]

    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'prediction':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

def convert_sync_batchnorm(module):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if hvd.rank()==0:
                print('Convert batchnorm: ',module)
            #print(SyncBatchNorm)
            module_output = SyncBatchNorm(module.num_features, module.eps, module.momentum,module.affine,module.track_running_stats)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, convert_sync_batchnorm(child))
        del module
        #print('Convert batchnorm finished!')
        return module_output


Style.RESET_ALL

"""=========================INPUT ARGUMENTS====================="""

parser = argparse.ArgumentParser()

parser.add_argument('--split',        default=-1,   type=int, help='Train/test classes split. Use -1 for kinetics2ucf')
parser.add_argument('--dataset',      default='kinetics2both',   type=str, help='Dataset: kinetics2both')

parser.add_argument('--train_samples',  default=-1,  type=int, help='Reduce number of train samples to the given value')
parser.add_argument('--class_total',  default=-1,  type=int, help='For debugging only. Reduce the total number of classes')

parser.add_argument('--clip_len',     default=16,   type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips',     default=1,   type=int, help='Number of clips per video')

parser.add_argument('--class_overlap', default=0.05,  type=float, help='tau. see Eq.3 in main paper')

### General Training Parameters
parser.add_argument('--lr',           default=0.1, type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs',     default=150,   type=int,   help='Number of training epochs.')
parser.add_argument('--bs',           default=1,   type=int,   help='Mini-Batchsize size per GPU.')
parser.add_argument('--size',         default=112,  type=int,   help='Image size in input.')

parser.add_argument('--fixconvs', action='store_true', default=False,   help='Freezing conv layers')
parser.add_argument('--nopretrained', action='store_false', default=True,   help='Pretrain network.')

##### Network parameters
parser.add_argument('--network', default='r2plus1d_18', type=str,
                    help='Network backend choice: [resnet18, r2plus1d_18, r3d_18, c3d].')

### Paths to datasets and storage folder
parser.add_argument('--save_path',    default='/workspace/debug/', type=str, help='Where to save log and checkpoint.')
parser.add_argument('--weights',      default=None, type=str, help='Weights to load from a previously run.')
parser.add_argument('--progressbar', action='store_true', default=False,   help='Show progress bar during train/test.')
parser.add_argument('--evaluate', action='store_true', default=False,   help='Evaluation only using 25 clips per video')
parser.add_argument('--wordsmodel',    default='/workspace/word2vec/GoogleNews-vectors-negative300.bin', type=str, help='words model')
parser.add_argument('--ucf101path',    default='data/ucf101_id_exist.txt', type=str, help='ucf101 path')
parser.add_argument('--hmdb51path',    default='data/hmdb51_id_exist.txt', type=str, help='hmdb51path')
parser.add_argument('--kinetics700path',    default='data/kinetics_id_exist.txt', type=str, help='kinetics700path')
parser.add_argument('--lmdbpath',    default='/workspace/action_lmdbfiles/lmdb/', type=str, help='lmdbpath')
parser.add_argument('--kernels',    default=4, type=int, help='kernels')
parser.add_argument('--solution',    default=0, type=int, help='solution')
parser.add_argument('--resume',    default=1, type=int, help='resume')
##### Read in parameters
opt = parser.parse_args()

opt.multiple_clips = False

torch.cuda.set_device(hvd.local_rank())
torch.set_num_threads(opt.kernels)


"""=================================DATALOADER SETUPS====================="""
Total_batch_size = opt.bs * hvd.size()
if hvd.rank() == 0:
    print('Single and Total batch size: %d, %d' % (opt.bs, Total_batch_size))
dataloaders = dataset.get_dataloaders(opt,hvd)
if not opt.evaluate:
    opt.n_classes = dataloaders['training'][0].dataset.class_embed.shape[0]
else:
    opt.n_classes = dataloaders['testing'][0].dataset.class_embed.shape[0]

"""=================================OUTPUT FOLDER====================="""
opt.savename = opt.save_path + '/'
if not opt.evaluate:
    opt.savename += '%s/CLIP%d_LR%f_%s_BS%d' % (
            opt.dataset, opt.clip_len,
            opt.lr, opt.network, opt.bs)

    if opt.class_overlap > 0:
        opt.savename += '_CLASSOVERLAP%.2f' % opt.class_overlap

    if opt.class_total != -1:
        opt.savename += '_NCLASS%d' % opt.class_total

    if opt.train_samples != -1:
        opt.savename += '_NTRAIN%d' % opt.train_samples

    if opt.fixconvs:
        opt.savename += '_FixedConvs'

    if not opt.nopretrained:
        opt.savename += '_NotPretrained'
    if opt.solution == 0:
       opt.savename += '_ours'
    if opt.split != -1:
        opt.savename += '/split%d' % opt.split

else:
    opt.weights = opt.savename + 'checkpoint.pth.tar'
    opt.savename += '/evaluation/'

if hvd.rank() == 0:
    if not os.path.exists(opt.savename+'/samples/'):
        os.makedirs(opt.savename+'/samples/')

"""=============================NETWORK SETUP==============================="""
img_model = network.get_network(opt)
words_model = network.Backbone_words()
model = network.AURL(img_model, words_model)
model = convert_sync_batchnorm(model)
model  = model.cuda()

if opt.weights and opt.weights != "none":
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['state_dict'])
    if hvd.rank() == 0:
       print("LOADED MODEL:  ", opt.weights, 'accuracy: ', checkpoint['accuracy'])

"""==========================OPTIM SETUP=================================="""
optimizer = get_optimizer(
        'sgd', model,
        lr=opt.lr*Total_batch_size/512,
        momentum=0.9,
        weight_decay=0.0001)
num_iters = 100000//Total_batch_size
scheduler = LR_Scheduler(
        optimizer,
        1, 0.0,
        int(opt.n_epochs), opt.lr*Total_batch_size/512,
        0.0,
        num_iters,
        constant_predictor_lr=False)

opt.start_epoch = -1
if opt.resume==1 and os.path.exists(opt.savename + '/last_checkpoint.pth.tar'):
    checkpoint = torch.load(opt.savename + '/last_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    opt.start_epoch = checkpoint['epoch']
    scheduler.iter = checkpoint['schedular_step']
    if hvd.rank() == 0:
        print("LOADED MODEL:  ", opt.savename + '/last_checkpoint.pth.tar', 'lr: ',optimizer.param_groups[0]['lr'], 'start_epoch: ', opt.start_epoch, 'scheduler.iter: ', scheduler.iter )
    

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

"""===========================TRAINER FUNCTION==============================="""

def train_one_epoch(train_dataloader, model, optimizer, opt, epoch, total_batch):
    class_embedding = train_dataloader.dataset.class_embed
    class_names = train_dataloader.dataset.class_name
    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader
    data_iterator.sampler.set_epoch(epoch+hvd.rank())

        
    log_str = '########################### Epoch: %d ###########################' % (epoch)
    if hvd.rank() == 0:
        print(log_str)
    for i, (X, l, Z, _) in enumerate(data_iterator):
        if i==num_iters:
            break
        not_broken = l != -1
        X, l, Z = X[not_broken], l[not_broken], Z[not_broken]
        batch_times.append(time.time() - tt_batch)
        s = list(X.shape)

        tt_model = time.time()
        loss, Y, _ = model(X.cuda(),torch.from_numpy(class_embedding).cuda(),l.cuda(), opt=opt)
        pred_embed = Y.detach().cpu().numpy()
        
        with torch.no_grad():
            class_embedding_new = model.encode_words(torch.from_numpy(class_embedding).cuda())
            class_embedding_new = class_embedding_new.cpu().detach().numpy()
        
        pred_label = cdist(pred_embed, class_embedding_new, 'cosine').argmin(1)
        acc = accuracy_score(l.numpy(), pred_label) * 100
        accuracy_regressor.append(acc)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        model_times.append(time.time() - tt_model)
        losses.append(loss.item())
        if hvd.rank() == 0:
            log_str = 'Epoch: {0:>5}, Iter: {1:>5}, Loss: {2:>7.5},Loss_avg: {3}, lr: {4}, batch_size: {5}, Total_batch_size: {6}, num_iters: {7}, constant_ls: {8}, acc:{9},acc_avg:{10}'\
                          .format(epoch, i, loss.item(), np.mean(losses), optimizer.param_groups[0]['lr'], s[0], Total_batch_size, num_iters, 0,acc,np.mean(accuracy_regressor))
            print(log_str)

        tt_batch = time.time()
        total_batch = total_batch + 1
    if hvd.rank() == 0:
        batch_times, model_times = np.sum(batch_times), np.sum(model_times)
        print('TOTAL time for: load the batch %.2f sec, run the model %.2f sec, train %.2f min' % (
                                    batch_times, model_times, (batch_times+model_times)/60))
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optim_state_dict': optimizer.state_dict(), 'schedular_step': scheduler.iter},
                       opt.savename + '/last_checkpoint.pth.tar')
    return total_batch


"""========================================================="""


def evaluate(test_dataloader, epoch):
    name = test_dataloader.dataset.name
    _ = model.eval()
    with torch.no_grad():
        ### For all test images, extract features
        n_samples = len(test_dataloader.dataset)

        predicted_embed = np.zeros([n_samples, 2048], 'float32')
        true_embed = np.zeros([n_samples, 2048], 'float32')
        true_label = np.zeros(n_samples, 'int')
        good_samples = np.zeros(n_samples, 'int') == 1

        final_iter = test_dataloader

        fi = 0
        for idx, data in enumerate(final_iter):
            X, l, Z, _ = data
            not_broken = l != -1
            X, l, Z = X[not_broken], l[not_broken], Z[not_broken]
            if len(X) == 0: continue
            # Run network on batch
            class_embedding = test_dataloader.dataset.class_embed
            _, Y, Z = model(X.cuda(),torch.from_numpy(class_embedding).cuda(),l.cuda())
            Y = Y.cpu().detach().numpy()
            Z = Z.cpu().detach().numpy()
            l = l.cpu().detach().numpy()
            Z = Z[l]
            predicted_embed[fi:fi + len(l)] = Y
            true_embed[fi:fi + len(l)] = Z.squeeze()
            true_label[fi:fi + len(l)] = l.squeeze()
            good_samples[fi:fi + len(l)] = True
            fi += len(l)

    predicted_embed = predicted_embed[:fi]
    true_embed, true_label = true_embed[:fi], true_label[:fi]

    # Calculate accuracy over test classes
    class_embedding = test_dataloader.dataset.class_embed
    
    with torch.no_grad():
        class_embedding = model.encode_words(torch.from_numpy(class_embedding).cuda())
        class_embedding = class_embedding.cpu().detach().numpy()
   
    accuracy, accuracy_top5 = compute_accuracy(predicted_embed, class_embedding, true_embed)

    log_str = 'Epoch: {0:>5}, accuracy: {1}, accuracy_top5: {2}, Name: {3}'.format(epoch, accuracy, accuracy_top5, name.upper())
    if hvd.rank() == 0:
        print(log_str)


    # Logging accuracy in CSV file
    if hvd.rank() == 0:
        with open(opt.savename+'/'+name+'_accuracy.csv', 'a') as f:
            f.write('%d, %.1f,%.1f\n' % (epoch, accuracy, accuracy_top5))
    '''
    if opt.split == -1:
        # Calculate accuracy per split
        # Only when the model has been trained on a different dataset
        accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
        for split in range(len(accuracy_split)):
            # Select test set
            np.random.seed(split) # fix seed for future comparability
            sel_classes = np.random.permutation(len(class_embedding))[:len(class_embedding) // 2]
            sel = [l in sel_classes for l in true_label]
            test_classes = len(sel_classes)

            # Compute accuracy
            subclasses = np.unique(true_label[sel])
            tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])
            acc, acc5 = compute_accuracy(predicted_embed[sel], class_embedding[sel_classes], true_embed[sel])
            accuracy_split[split] = acc
            accuracy_split_top5[split] = acc5

        # Printing on terminal
        res_str += ' -- Split accuracy %2.1f%% (+-%.1f) on %d classes' % (
                        accuracy_split.mean(), accuracy_split.std(), test_classes)
        accuracy_split, accuracy_split_std = np.mean(accuracy_split), np.std(accuracy_split)
        accuracy_split_top5, accuracy_split_top5_std = np.mean(accuracy_split_top5), np.std(accuracy_split_top5)

        # Logging using tensorboard
        
        txwriter.add_scalar(name+'/AccSplit_Mean', accuracy_split, epoch)
        txwriter.add_scalar(name+'/AccSplit_Std', accuracy_split_std, epoch)
        txwriter.add_scalar(name+'/AccSplit_Mean_Top5', accuracy_split_top5, epoch)
        txwriter.add_scalar(name+'/AccSplit_Std_Top5', accuracy_split_top5_std, epoch)
        

        # Logging accuracy in CSV file
        with open(opt.savename + '/' + name + '_accuracy_splits.csv', 'a') as f:
            f.write('%d, %.1f,%.1f,%.1f,%.1f\n' % (epoch, accuracy_split, accuracy_split_std,
                                                   accuracy_split_top5, accuracy_split_top5_std))
    
    print(Fore.GREEN, res_str, Style.RESET_ALL)
    '''
    return accuracy, accuracy_top5


def compute_accuracy(predicted_embed, class_embed, true_embed):
    assert len(predicted_embed) == len(true_embed), "True and predicted labels must have the same number of samples"
    y_pred = cdist(predicted_embed, class_embed, 'cosine').argsort(1)
    y = cdist(true_embed, class_embed, 'cosine').argmin(1)
    accuracy = accuracy_score(y, y_pred[:, 0]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, :5])]) * 100
    return accuracy, accuracy_top5


"""===================SCRIPT MAIN========================="""

if __name__ == '__main__':
    total_batch = 0
    trainsamples = 0
    if not opt.evaluate:
        trainsamples = len(dataloaders['training'][0].dataset)
        if hvd.rank() == 0:
            with open(opt.savename + '/train_samples_%d_%d.txt' % (opt.n_classes, trainsamples), 'w') as f:
                f.write('%d, %d\n' % (opt.n_classes, trainsamples) )

    best_acc = 0
    if hvd.rank() == 0:
        print('\n----------')
    #txwriter = SummaryWriter(logdir=opt.savename)
    epoch_times = []
    for epoch in range(opt.start_epoch+1, opt.n_epochs):
        if hvd.rank() == 0:
            print('\n{} classes {} from {}, LR {} BS {} CLIP_LEN {} N_CLIPS {} OVERLAP {} SAMPLES {}'.format(
                    opt.network.upper(), opt.n_classes,
                    opt.dataset.upper(), opt.lr, opt.bs, opt.clip_len, opt.n_clips,
                    opt.class_overlap, trainsamples))
            print(opt.savename)
        tt = time.time()

        ## Train one epoch
        if not opt.evaluate:
            _ = model.train()
            total_batch = train_one_epoch(dataloaders['training'][0], model, optimizer, opt, epoch, total_batch)

        ### Evaluation
        accuracies = []
        for test_dataloader in dataloaders['testing']:
            accuracy, _ = evaluate(test_dataloader, epoch)
            accuracies.append(accuracy)
        accuracy = np.mean(accuracies)

        if accuracy > best_acc:
            # Save best model
            if hvd.rank() == 0:
                torch.save({'state_dict': model.state_dict(), 'opt': opt, 'accuracy': accuracy},
                       opt.savename + '/checkpoint.pth.tar')
            best_acc = accuracy

        epoch_times.append(time.time() - tt)
        if hvd.rank() == 0:
            print('----- Epoch ', Fore.RED, '%d' % epoch, Style.RESET_ALL,
              'done in %.2f minutes. Remaining %.2f minutes.' % (
              epoch_times[-1]/60, ((opt.n_epochs-epoch-1)*np.mean(epoch_times))/60),
              Fore.BLUE, 'Best accuracy %.1f' % best_acc, Style.RESET_ALL)
        #scheduler.step()
        opt.lr = optimizer.param_groups[0]['lr']

        if opt.evaluate:
            break


