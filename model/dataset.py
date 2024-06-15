from utils import *
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms.functional as F
import torchvision as t
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import numpy as np
# import albumentations as A
# import cv2
# class Target_Augmentation1(object):
   
#     # Based on Random Erasing
    
#     def __init__(self,  sl = 0.02, sh = 0.4, r1 = 0.3, numbers = 2): 
        
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#         self.target_numbers = numbers
        
#     def __call__(self, img, mask):
#         img = np.expand_dims(img, axis=0)
#         img= np.tile(img, (3, 1, 1))
#         # print(img.shape)

#         area = 9 * 9
 
#         w = []
#         h = []
        
#         target_area = random.uniform(self.sl, self.sh) * area  
#         aspect_ratio = random.uniform(self.r1, 1/self.r1)
        
#         for i in range(self.target_numbers):
            
#             h.append(int(round(math.sqrt(target_area * aspect_ratio))))
#             w.append(int(round(math.sqrt(target_area / aspect_ratio)))) 
            
#         img_aug = np.array(img)    
#         mask_aug = np.array(mask)
        
#         # img_aug = img_aug.swapaxes(0,2)
#         # mask_aug = mask_aug.swapaxes(0,1)

#         x = []
#         y = []
        
#         # print(img_aug.shape)
#         if img_aug.shape[0]== 3:
            
#             for i in range(self.target_numbers):

#                 if w[i] < img_aug.shape[1] and h[i] < img_aug.shape[2]:
                    
#                     x.append(random.randint(0, img_aug.shape[1] - h[i])) 
#                     y.append(random.randint(0, img_aug.shape[2] - w[i]))
                
#                 if img_aug[0,x[i],y[i]] > 0:
                            
#                         m1, m2 = np.mgrid[:h[i], :w[i]]
#                         target = (m2 - h[i]//2) ** 2 + (m1 - w[i]//2) ** 2
                        
#                         target = -target + np.mean((img_aug)[0])
#                         target[target < np.min((img_aug)[0])] = np.min((img_aug)[0])
#                         theta = int(np.min((img_aug)[0])/np.max((img_aug)[0]) * np.mean((img_aug)[0]))
#                         target[target >= theta] = np.max((img_aug)[0])

#                         hh = target.shape[0]
#                         ww = target.shape[1]
                        
#                         img_aug[0, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
#                         img_aug[1, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
#                         img_aug[2, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
#                         mask_aug[x[i]:x[i]+hh, y[i]:y[i]+ww] = 255

#         # img = img_aug.swapaxes(0,2)
#         # mask = mask_aug.swapaxes(0,1)
#         img = img_aug
#         mask = mask_aug
        
#         return img[0], mask

# class Target_Augmentation2(object):
   
#     # Based on Random Erasing
    
#     def __init__(self,  sl = 0.02, sh = 0.4, r1 = 0.3, numbers = 2): 
        
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#         self.target_numbers = numbers
        
#     def __call__(self, img, mask):
#         img = np.expand_dims(img, axis=0)
#         img= np.tile(img, (3, 1, 1))
#         # print(img.shape)

#         area = 10 * 10
 
#         w = []
#         h = []
        
#         target_area = random.uniform(self.sl, self.sh) * area  
#         aspect_ratio = random.uniform(self.r1, 1/self.r1)
        
#         for i in range(self.target_numbers):
            
#             h.append(int(round(math.sqrt(target_area * aspect_ratio))))
#             w.append(int(round(math.sqrt(target_area / aspect_ratio)))) 
            
#         img_aug = np.array(img)    
#         mask_aug = np.array(mask)
        
#         # img_aug = img_aug.swapaxes(0,2)
#         # mask_aug = mask_aug.swapaxes(0,1)

#         x = []
#         y = []
        
#         # print(img_aug.shape)
#         if img_aug.shape[0]== 3:
            
#             for i in range(self.target_numbers):

#                 if w[i] < img_aug.shape[1] and h[i] < img_aug.shape[2]:
                    
#                     x.append(random.randint(0, img_aug.shape[1] - h[i])) 
#                     y.append(random.randint(0, img_aug.shape[2] - w[i]))
                
#                 if img_aug[0,x[i],y[i]] > 0:
                            
#                         m1, m2 = np.mgrid[:h[i], :w[i]]
#                         target = (m2 - h[i]//2) ** 2 + (m1 - w[i]//2) ** 2
                        
#                         target = -target + np.mean((img_aug)[0])
#                         # target[target < np.min((img_aug)[0])] = np.min((img_aug)[0])
#                         target[target < np.mean((img_aug)[0])] = np.mean((img_aug)[0])
#                         # theta = int(np.min((img_aug)[0])/np.max((img_aug)[0]) * np.mean((img_aug)[0]))
#                         # target[target >= np.min((img_aug)[0])] = np.max((img_aug)[0])
#                         target[target >= np.mean((img_aug)[0])] = np.max((img_aug)[0])

#                         hh = target.shape[0]
#                         ww = target.shape[1]
                        
#                         img_aug[0, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
#                         img_aug[1, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
#                         img_aug[2, x[i]:x[i]+hh, y[i]:y[i]+ww] = target
#                         mask_aug[x[i]:x[i]+hh, y[i]:y[i]+ww] = 255

#         # img = img_aug.swapaxes(0,2)
#         # mask = mask_aug.swapaxes(0,1)
#         img = img_aug
#         mask = mask_aug
        
#         return img[0], mask

# def copy_paste_augmentation(image, mask, target_threshold=5, scale_range=(0.6, 0.9)):
#     """
#     对输入图像进行copy-paste增强。
    
#     参数:
#     - image: 输入图像
#     - mask: 目标掩码
#     - target_threshold: 目标数量阈值，小于该值时进行copy-paste
#     - scale_range: 缩放范围

#     返回:
#     - augmented_image: 增强后的图像
#     - augmented_mask: 增强后的掩码
#     """
#     augmented_image = image.copy()
#     augmented_mask = mask.copy()
    
#     # 找到所有目标的连通区域
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
#     if num_labels-1 == 0:
#         return augmented_image, augmented_mask
#     # 如果目标数量小于 target_threshold，则复制某个目标两次
#     if num_labels - 1 < target_threshold:
#         target_idx = random.choice(range(1, num_labels))
        
#         for _ in range(3):  # 复制两次
#             # 提取目标块
#             ys, xs = np.where(labels == target_idx)
#             target_image_patch = image[min(ys):max(ys)+1, min(xs):max(xs)+1]
#             target_mask_patch = mask[min(ys):max(ys)+1, min(xs):max(xs)+1]
            
#             # 随机缩放
#             scale = random.uniform(scale_range[0], scale_range[1])
#             target_image_patch = cv2.resize(target_image_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#             target_mask_patch = cv2.resize(target_mask_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
#             # 随机粘贴位置
#             paste_y = random.randint(0, image.shape[0] - target_image_patch.shape[0])
#             paste_x = random.randint(0, image.shape[1] - target_image_patch.shape[1])
            
#             # 粘贴目标块
#             patch_ys, patch_xs = np.where(target_mask_patch > 0)
#             for dy, dx in zip(patch_ys, patch_xs):
#                 augmented_image[paste_y + dy, paste_x + dx] = target_image_patch[dy, dx]
#                 augmented_mask[paste_y + dy, paste_x + dx] = target_mask_patch[dy, dx]
    
#     return augmented_image, augmented_mask

# class TrainSetLoader(Dataset):
#     def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None,fold=0):
#         super(TrainSetLoader).__init__()
#         self.dataset_name = dataset_name
#         self.dataset_dir = dataset_dir + '/' + dataset_name
#         self.patch_size = patch_size
#         self.fold = fold
#         # with open(self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
#         with open(self.dataset_dir +'/img_idx/train.txt', 'r') as f:
#         # with open(f'/home/dww/OD/BasicIRSTD/train_fold{self.fold}.txt', 'w') as f:
#             self.train_list = f.read().splitlines()
#         if img_norm_cfg == None:
#             self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
#         else:
#             self.img_norm_cfg = img_norm_cfg
#         self.tranform = augumentation()
#         self.transform2 = A.Compose([  ####其他增强
#             A.HorizontalFlip(p=0.5),
#             # A.RandomBrightnessContrast(p=0.2),
#             A.VerticalFlip(p=0.5),
#             A.Rotate(limit=30, p=0.5),
#             # A.RandomBrightnessContrast(p=0.5),
#             # A.HueSaturationValue(p=0.5),
#             # A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#             # A.GaussNoise(p=0.5),
#             # A.MultiplicativeNoise(p=0.5),
#         ])
        
#     def __getitem__(self, idx):
#         try:
#             img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/')).convert('I')
#             mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//','/')).convert('L')
#             # imag = np.array(img).astype(np.uint8)
#             # mas = np.array(mask).astype(np.uint8)
#             img,mask = copy_paste_augmentation(np.array(img).astype(np.uint8), np.array(mask).astype(np.uint8)) ###copypaste 增强
#             # img = Image.fromarray(img)
#             # mask = Image.fromarray(mask)
#             # img,mask = self.transform2(image=imag, mask=mas)
#             # img = Image.fromarray(np.uint8(img))
#             # mask = Image.fromarray(np.uint8(mask))
#             # plt.imshow(mask)
#             # plt.savefig('mask.jpg', dpi=400)
#             # transformed = self.transform2(image=np.array(img).astype(np.uint8), mask=np.array(mask).astype(np.uint8)) ###增强
#             # newimg = transformed['image']
#             # newmask = transformed['mask']
#             img , mask = Target_Augmentation2()(img, mask)
            
#             # plt.imshow(mask)
#             # plt.savefig('mask2.jpg', dpi=400)
#         except:
#             img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//','/')).convert('I')
#             mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//','/'))
#         img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
#         mask = np.array(mask, dtype=np.float32)  / 255.0
#         if len(mask.shape) > 2:
#             mask = mask[:,:,0]
        
        
#         img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
#         img_patch, mask_patch = self.tranform(img_patch, mask_patch)

#         img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
#         img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
#         mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
#         return img_patch, mask_patch
#     def __len__(self):
#         return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None,fold=0):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.fold = fold
        # with open(self.dataset_dir + '/img_idx/test.txt', 'r') as f:
        # with open(f'/home/dww/OD/BasicIRSTD/val_fold{fold}.txt', 'w') as f:
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/')).convert('L')
            ###resize下
            # img = img.resize((512, 512))
            # mask = mask.resize((512, 512))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))

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
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

# class EvalSetLoader(Dataset):
#     def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
#         super(EvalSetLoader).__init__()
#         self.dataset_dir = dataset_dir
#         self.mask_pred_dir = mask_pred_dir
#         self.test_dataset_name = test_dataset_name
#         self.model_name = model_name
#         with open(self.dataset_dir+'/img_idx/val' + test_dataset_name + '.txt', 'r') as f:
#             self.test_list = f.read().splitlines()

#     def __getitem__(self, idx):
#         mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//','/'))
#         mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

#         mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
#         mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
#         if len(mask_pred.shape) == 3:
#             mask_pred = mask_pred[:,:,0]
        
#         h, w = mask_pred.shape
        
#         mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
#         mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
#         mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
#         return mask_pred, mask_gt, [h,w]
#     def __len__(self):
#         return len(self.test_list) 

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

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
