import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from model.dataset import *
import matplotlib.pyplot as plt
from metric import *
import os
import time
from sklearn.metrics import  auc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ACM'], type=list, 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['NUAA-SIRST/ACM_400.pth.tar'], type=list, help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='/home/dww/OD/dataset', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='/home/dww/OD/BasicIRSTD/results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='/home/dww/OD/BasicIRSTD/log_seed_posSample/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()
    
    # eval_mIoU = mIoU() 
    # eval_PD_FA = PD_FA()
    # eval_roc = ROCMetric(1,10)
    # metric = SegmentationMetricTPFNFP(nclass=1)
    t_all = []
    
    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        start = time.time()
        img = Variable(img).cuda()
        pred = net.forward(img)
        # pred = pred[:,:,:size[0],:size[1]]
        # gt_mask = gt_mask[:,:,:size[0],:size[1]]
        # eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        # eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
        # eval_roc.update(pred[0,0,:,:],gt_mask[0,0,:,:])
        # metric.update(labels=gt_mask, preds=(pred>opt.threshold).cpu())
        end = time.time()
        t_all.append(end-start)
    
    print('average time:', np.mean(t_all) / 1)
    print('average fps:',1 / np.mean(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:',1 / min(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:',1 / max(t_all))

    opt.f.write('average time:'+ str(np.mean(t_all) / 1) + '\n')
    opt.f.write('average fps:'+ str(1 / np.mean(t_all))+ '\n')
    opt.f.write('fastest time:'+ str(min(t_all) / 1)+ '\n')
    opt.f.write('fastest fps:'+ str(1 / min(t_all))+ '\n')
    opt.f.write('slowest time:'+ str(max(t_all) / 1)+ '\n')
    opt.f.write('slowest fps:'+ str(1 / max(t_all))+ '\n')


    # tp_rates, fp_rates = eval_roc.get()
    # # miou, prec, recall, fscore
    # # iou,F1,recall,precision = metric.get() 12月3号计算的NUAA数据集准确率和F1反了
    # iou,precision,recall,F1 = metric.get()
    # AUC = auc(fp_rates,tp_rates)    
    # results1 = eval_mIoU.get()
    # results2 = eval_PD_FA.get()
    # print("iou:\t" + str(iou))
    # print("pixAcc, mIoU:\t" + str(results1))
    # print("PD, FA, nIoU:\t" + str(results2))
    # print("AUC:\t"+str(AUC))
    # print("F1:\t"+str(F1))
    # print("precision:\t"+str(precision))
    # print("recall:\t"+str(recall))
    # opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    # opt.f.write("PD, FA,niou:\t" + str(results2) + '\n')
    # opt.f.write("AUC:\t" + str(AUC) + '\n')
    # opt.f.write("F1:\t" + str(F1) + '\n')
    # opt.f.write("precision:\t" + str(precision) + '\n')
    # opt.f.write("recall:\t" + str(recall) + '\n')
    # opt.f.write('tpr'+'\n')
    # for i in range(len(tp_rates)):
    #     opt.f.write('   ')
    #     opt.f.write(str(round(tp_rates[i], 4)))
    #     opt.f.write(' ,  ')
    # opt.f.write('\n'+'fpr'+'\n')
    # for i in range(len(fp_rates)):
    #     opt.f.write('   ')
    #     opt.f.write(str(round(fp_rates[i], 4)))
    #     opt.f.write('  , ')
    # opt.f.write('\n') 
    # eval_roc.reset()
    # metric.reset()
    # eval_mIoU.reset()
    # eval_PD_FA.reset()

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'calfps_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()

            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
        
