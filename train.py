import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from model.dataset import *
import matplotlib.pyplot as plt
from metric import *
import numpy as np
import os
import os.path as osp
from utils import *
from sklearn.metrics import auc
import torch.utils.data as Data
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['AGPCNet'], type=list, 
                    help="model_name: 'ACM', 'ALCNet', 'RPCANet','DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")              
parser.add_argument("--dataset_names", default=['PRCV'], type=list, 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='/home/public', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=3407, help="Threshold for test")

global opt
opt = parser.parse_args()

seed_pytorch(opt.seed)
class MDFADataset(Data.Dataset):
    def __init__(self, base_dir='/home/dww/OD/dataset/mDFA', mode='train', base_size=256):
        assert mode in ['train', 'test']
        self.base_size = base_size
        self.mode = mode
        if mode == 'train':
            self.img_dir = osp.join(base_dir, 'training')
            self.mask_dir = osp.join(base_dir, 'training')
        elif mode == 'test':
            self.img_dir = osp.join(base_dir, 'test_org')
            self.mask_dir = osp.join(base_dir, 'test_gt')
        else:
            raise NotImplementedError
        self.tranform = augumentation()
        self.img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    
    def __getitem__(self, i):
        if self.mode == 'train':
            img_path = osp.join(self.img_dir, '%06d_1.png' % i)
            mask_path = osp.join(self.mask_dir, '%06d_2.png' % i)
            try:
                img = Image.open(img_path).convert('I')
                mask = Image.open(mask_path)
                img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
                mask = np.array(mask, dtype=np.float32)  / 255.0
                if len(mask.shape) > 2:
                 mask = mask[:,:,0]
                img_patch, mask_patch = random_crop(img, mask, self.base_size, pos_prob=0.5) 
                img_patch, mask_patch = self.tranform(img_patch, mask_patch)
                img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
                img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
                mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
                return img_patch, mask_patch
            except (OSError,SyntaxError) as e:
                print(img_path)
        elif self.mode == 'test':
            img_path = osp.join(self.img_dir, '%05d.png' % i)
            mask_path = osp.join(self.mask_dir, '%05d.png' % i)
            img = Image.open(img_path).convert('I')
            mask = Image.open(mask_path)
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
        
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
        
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h,w], 0
        else:
            raise NotImplementedError
        
    def __len__(self):
        if self.mode == 'train':
            return 9978
        elif self.mode == 'test':
            return 100
        else:
            raise NotImplementedError

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
def train():
    if opt.dataset_name == 'MDFA':
        train_set = MDFADataset(mode='train', base_size=opt.patchSize)
        
    else:
        train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    
    net = Net(model_name=opt.model_name, mode='train').cuda()
    net.train()
    
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    
    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
    
    ### Default settings                
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs':400, 'step': [150, 300], 'gamma': 0.1}
    
    ### Default settings of DNANet                
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs':1500, 'min_lr':1e-5}
        
    opt.nEpochs = opt.scheduler_settings['epochs']
        
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)
    
    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1:
                continue
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask,data=img)
            total_loss_epoch.append(loss.detach().cpu().item()) #RPCAloss 加上item
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(total_loss_epoch)
        scheduler.step()
        if (idx_epoch + 1) % 10 == 0:
            
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            
        if (idx_epoch + 1) % 2 == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            test(save_pth)
            
        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            test(save_pth)

def test(save_pth):
    if opt.dataset_name == 'MDFA':
        test_set = MDFADataset(mode='test', base_size=opt.patchSize)
    else: 
        test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    eval_roc = ROCMetric(1,10)
    metric = SegmentationMetricTPFNFP(nclass=1)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
            img = Variable(img).cuda()
            pred = net.forward(img)
            pred = pred[:,:,:size[0],:size[1]]
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
            eval_roc.update(pred[0,0,:,:],gt_mask[0,0,:,:])
            metric.update(labels=gt_mask, preds=(pred>opt.threshold).cpu())
       

    tp_rates, fp_rates = eval_roc.get()
   
    # iou,F1,recall,precision = metric.get()
    iou,precision,recall,F1 = metric.get()
    AUC = auc(fp_rates,tp_rates)    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
 
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA, niou:\t" + str(results2))
    print("AUC:\t"+str(AUC))
    print("F1:\t"+str(F1))
    print("precision:\t"+str(precision))
    print("recall:\t"+str(recall))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA,niou:\t" + str(results2) + '\n')
    opt.f.write("AUC:\t" + str(AUC) + '\n')
    opt.f.write("F1:\t" + str(F1) + '\n')
    opt.f.write("precision:\t" + str(precision) + '\n')
    opt.f.write("recall:\t" + str(recall) + '\n')
    opt.f.write('tpr'+'\n')
    for i in range(len(tp_rates)):
        opt.f.write('   ')
        opt.f.write(str(round(tp_rates[i], 6)))
        opt.f.write(' ,  ')
    opt.f.write('\n')
    opt.f.write('fpr'+'\n')
    for i in range(len(fp_rates)):
        opt.f.write('   ')
        opt.f.write(str(round(fp_rates[i], 6)))
        opt.f.write('  , ')
    opt.f.write('\n')
    
    eval_roc.reset()
    metric.reset()
    eval_mIoU.reset()
    eval_PD_FA.reset()
    

def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
