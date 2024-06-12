import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from model.dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
import torch.nn.functional as F
from utils import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['ACM'], type=list, 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")              
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
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="5e-4optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument('--multi-gpus', type=bool, default=False)

global opt
opt = parser.parse_args()

seed_pytorch(opt.seed)

miou = 0
pd = 0
metric = 0
best_metric = 0


def Batch_Augmentation1 (img, mask): 
    
    Random_coefficient = random.randint(0, img.shape[0]-1) 
        
    img_s = []
    mask_s = []
        
    for i in range(img.shape[0]):

        img_s.append(img[i:i+1])
        mask_s.append(mask[i:i+1])

    data_aug = []
    label_aug = []
        
    for i in range(img.shape[0]):
            
        if Random_coefficient > img.shape[0]//2-1:
        
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),3))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),3))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),3))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),3))
                    
        else:
                
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),2))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),2))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),2))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),2))

    data = torch.cat(data_aug, dim=0)
    label = torch.cat(label_aug, dim=0)

    img = F.interpolate(data, size=[512, 512]) 
    mask = F.interpolate(label, size=[512, 512]) 
        
    # data = torch.cat((img,data),0)
    # label = torch.cat((mask,label),0)

    return img, mask

def Batch_Augmentation2 (img, mask): 

    
    Random_coefficient = random.randint(0, img.shape[0]-1) 
        
    img_s = []
    mask_s = []
        
    for i in range(img.shape[0]):

        img_s.append(img[i:i+1])
        mask_s.append(mask[i:i+1])

    data_aug = []
    label_aug = []
        
    for i in range(img.shape[0]):
            
        if Random_coefficient > img.shape[0]//2-1:
        
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),3))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),3))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),3))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),3))
                    
        else:
                
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),2))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),2))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),2))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),2))

    data = torch.cat(data_aug, dim=0)
    label = torch.cat(label_aug, dim=0)

    data = F.interpolate(data, size=[512, 512]) 
    label = F.interpolate(label, size=[512, 512]) 
        
    img = torch.cat((img,data),0)
    mask = torch.cat((mask,label),0)

    return img, mask
    
    

def train():

    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    
    device = torch.device('cuda')
    net = Net(model_name=opt.model_name, mode='train')
    if opt.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                net = nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)

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
        opt.scheduler_settings = {'epochs':400, 'step': [200, 300], 'gamma': 0.1}
    
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

            img, gt_mask = Batch_Augmentation2(img, gt_mask)

            pred = net.forward(img)
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if (idx_epoch + 1) % 1 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            
        if (idx_epoch + 1) % 1 == 0:
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
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    net = Net(model_name=opt.model_name, mode='test').cuda()

    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
            img = Variable(img).cuda()
            pred = net.forward(img)
            pred = pred[:,:,:size[0],:size[1]]
            # Resize pred to original image size
            pred = F.interpolate(pred, size=(size[0], size[1]), mode='bilinear', align_corners=True)
            #想把预测后的pred恢复到原图大小，但是不知道怎么恢复

            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    miou = results1[1]
    pd = results2[0]
    metric = 0.5 * (miou + pd)
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    print("metric:\t" + str(metric))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    opt.f.write("metric:\t" + str(metric) + '\n')
    
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
