import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from model.dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD Inference without mask")
parser.add_argument("--model_names", default=['ACM', 'ALCNet','DNANet', 'ISNet', 'RDIAN', 'ISTDU-Net'], nargs='+',  
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=None, nargs='+',  help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'], nargs='+', 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std
  
def test(): 
    test_set = InferenceSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    try:
        net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
    net.eval()

    # with torch.no_grad():
    #     for idx_iter, (img, size, img_dir) in tqdm(enumerate(test_loader)):
    #         img = Variable(img).cuda()
    #         pred = net.forward(img)
    #         pred = pred[:,:,:size[0],:size[1]]

    with torch.no_grad():
        max_block_size = (512, 512)
        for idx_iter, (img, size, img_dir) in tqdm(enumerate(test_loader)):
            img = Variable(img).cuda()
            _, _, height, width = img.size()

            if height < max_block_size[0] or width < max_block_size[1]:
                # 如果图像大小小于块的大小，不够形成完整的一块
                pred = net.forward(img)
            else:
                # 计算块的数量
                num_blocks_height = (height + max_block_size[0] - 1) // max_block_size[0]
                num_blocks_width = (width + max_block_size[1] - 1) // max_block_size[1]

                # 动态分块推理
                output = torch.zeros_like(img)
                for i in range(num_blocks_height):
                    for j in range(num_blocks_width):
                        # 计算当前块的位置和尺寸
                        block_y = i * max_block_size[0]
                        block_x = j * max_block_size[1]
                        block_height = min(max_block_size[0], height - block_y)
                        block_width = min(max_block_size[1], width - block_x)

                        # 提取当前块
                        block = img[:, :, block_y:block_y+block_height, block_x:block_x+block_width]

                        # 对当前块进行推理
                        pred_block = net.forward(block)

                        # 将结果放回输出的相应位置
                        output[:, :, block_y:block_y+block_height, block_x:block_x+block_width] = pred_block

                output = output[:, :, :size[0], :size[1]]
                pred = output        
            ### save img
            if opt.save_img == True:
                img_save = transforms.ToPILImage()(((pred[0,0,:,:]>opt.threshold).float()).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')  
    
    print('Inference Done!')
   
if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
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
        
