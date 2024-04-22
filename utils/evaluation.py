import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
import utils.datasets
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

def genertate_region_mask(masks, batch_shape):
    """generate B 1 H W meaningful-region-mask for a batch of masks

    Args:
        batch_shape (_type_): _description_
    """
    meaningful_mask = torch.zeros_like(masks)
    for idx, shape in enumerate(batch_shape):
        meaningful_mask[idx, :, :shape[0], :shape[1]] = 1
    return meaningful_mask

# def generate_original_mask(masks, batch_shape):
#     meaningful_mask = torch.zeros_like(masks)
#     for idx, shape in enumerate(batch_shape):
#         meaningful_mask = F.interpolate(mask)

def cal_confusion_matrix(predict, target, region_mask, threshold=0.5):
    """compute local confusion matrix for a batch of predict and target masks
    Args:
        predict (_type_): _description_
        target (_type_): _description_
        region (_type_): _description_
        
    Returns:
        TP, TN, FP, FN
    """
    predict = (predict > threshold).float()
    TP = torch.sum(predict * target * region_mask, dim=(1, 2, 3))
    TN = torch.sum((1-predict) * (1-target) * region_mask, dim=(1, 2, 3))
    FP = torch.sum(predict * (1-target) * region_mask, dim=(1, 2, 3))
    FN = torch.sum((1-predict) * target * region_mask, dim=(1, 2, 3))
    return TP, TN, FP, FN

def cal_F1(TP, TN, FP, FN):
    """compute F1 score for a batch of TP, TN, FP, FN
    Args:
        TP (_type_): _description_
        TN (_type_): _description_
        FP (_type_): _description_
        FN (_type_): _description_
    """
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    # F1 = torch.mean(F1) # fuse the Batch dimension
    return F1

def cal_pixel_f1_hzw(pred, target, shape, th=0.5):

    # print(pred.shape, target.shape)
    shape = shape[0]
    pred = pred[0,0,:shape[0],:shape[1]].cpu().numpy()
    
    # print(pred.shape, target.shape)
    # print(pred.shape, shape)
    # pred = pred[0:shape[0], 0:shape[1]]
    pred = (pred > th).astype(float)
    # target = target
    target = target[0,0,:shape[0],:shape[1]].cpu().numpy()
    # target = target[0:shape[0], 0:shape[1]]

    F1 = f1_score(y_true=target.flatten(), y_pred=pred.flatten(), zero_division='warn')
    return F1

def cal_pixel_f1(pred, target, th=0.5):

    # print(pred.shape, target.shape)
    # shape = shape[0]
    pred = pred[0,0,:,:].cpu().numpy()
    
    # print(pred.shape, target.shape)
    # print(pred.shape, shape)
    # pred = pred[0:shape[0], 0:shape[1]]
    pred = (pred > th).astype(float)
    # target = target
    target = target[0,0,:,:].cpu().numpy()
    # target = target[0:shape[0], 0:shape[1]]

    F1 = f1_score(y_true=target.flatten(), y_pred=pred.flatten(), zero_division='warn')
    return F1
    