import numpy as np
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

# from engine import *
import os
import sys
from sklearn.metrics import confusion_matrix
from utils import *
from configs.config_setting import setting_config

import cv2
import warnings

def iou(config, input, target):

    # preds = []
    # gts = []

    preds = [cv2.imread(os.path.join(input, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(input)]
    gts = [cv2.imread(os.path.join(target, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(target)]

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds>=0.4, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            # if test_data_name is not None:
            #     log_info = f'test_datasets_name: {test_data_name}'
            #     print(log_info)
            #     logger.info(log_info)
    log_info = f'test of best model,miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                    specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
    print(log_info)


if __name__=='__main__':
    input =r'/media/share02/fcp/packs/pycharm_project_907/VM-UNet-main/data/hou/val/outputs_zhanlian'
    target =r'/media/share02/fcp/packs/pycharm_project_907/VM-UNet-main/data/hou/val/masks'
    iou(setting_config, input, target)