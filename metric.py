import numpy as np
import torch
from skimage import measure
import threading

class SegmentationMetricTPFNFP(object):
    """Computes pixAcc and mIoU metric scroes
    """

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            tp, fp, fn = batch_tp_fp_fn(pred, label, self.nclass)
            with self.lock:
                self.total_tp += tp
                self.total_fp += fp
                self.total_fn += fn
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_tp, self.total_fp, self.total_fn

    def get(self):
        return get_miou_prec_recall_fscore(self.total_tp, self.total_fp, self.total_fn)

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        return



def get_miou_prec_recall_fscore(total_tp, total_fp, total_fn):
    miou = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
    prec = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp)
    recall = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fn)
    fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)

    return miou, prec, recall, fscore


def batch_tp_fp_fn(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """

    mini = 1
    maxi = nclass
    nbins = nclass

    predict = (output.detach().numpy() > 0).astype('int64')  # P
    target = target.numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0]
    area_fn = area_lab[0] - area_inter[0]

    # area_union = area_pred + area_lab - area_inter
    assert area_tp <= (area_tp + area_fn + area_fp)
    return area_tp, area_fp, area_fn


class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
        self.nIoU = 0
    def update(self, preds, labels, size):
        predits  = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64') 

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []
        self.Iou = 0

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)
                    intersection = np.sum(np.array(image==m+1)*np.array(label==i+1))
                    label_sum = np.sum(np.array(coord_label[i].area))
                    pred_sum = np.sum(area_image)
                    self.Iou += intersection/(label_sum+pred_sum-intersection)
                    del coord_image[m]
                    break
        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel +=np.sum(self.dismatch)
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)
        self.nIoU+=self.Iou

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target
        Final_nIoU = self.nIoU / self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy()),Final_nIoU

    def reset(self):
        # self.FA  = np.zeros([self.bins+1])
        # self.PD  = np.zeros([self.bins+1])
        self.FA  = 0
        self.PD  = 0

def batch_pix_accuracy(output, target):   
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

class ROCMetric():
    def __init__(self, nclass=1, bins=10):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值 
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.reset()
    """
    更新评估指标,preds是预测结果,labels是标签
    """
    def update(self, preds, labels):
        for iBin in range(self.bins+1): #循环10次
            score_thresh = (iBin + 0.0) / self.bins
           
            # print(iBin, "-th, score_thresh: ", score_thresh)

            i_tp, i_pos, i_fp, i_neg, i_class_pos = Cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):
        
        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)  ##### pos 代表真值为正，neg代表真值为负
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        # recall      = self.tp_arr / (self.pos_arr   + 0.001)
        # precision   = self.tp_arr / (self.class_pos + 0.001)  ###class_pos 代表被分类为正值的总数

        # recall = recall.mean()

        # precision = precision.mean()

        # F1 = 2 * (precision * recall) / (precision + recall)

        # AUC = auc(fp_rates,tp_rates)

        # False_alarm = (self.class_pos -self.tp_arr) / (self.pos_arr + self.neg_arr + 0.001) #错误分类为正的样本除以样本总数

        return tp_rates, fp_rates #### tp_rates 代表Recall 值， fp_rate 代表 false positive rate

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])

        # print('reset_self.tp_arr:', self.tp_arr)


def Cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = (torch.sigmoid(output).cpu().detach().numpy() > score_thresh).astype('int64') # P
    target = target.cpu().detach().numpy().astype('int64')  # T
    intersection = predict * (predict == target) # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp
    return tp, pos, fp, neg, class_pos