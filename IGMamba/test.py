import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.igmamba.igmamba import IGMambaNet

# from engine import *
import os
import sys
from sklearn.metrics import confusion_matrix
from utils import *
from configs.config_setting import setting_config

import cv2
import warnings
warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    model_path = r'/media/share02/fcp/packs/pycharm_project_907/IGMambaNet-main/results/hou_Monday_24_March_2025_16h_41m_27s'
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'best-epoch85-loss0.8450.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    # global writer

    # writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')


    test_dataset = NPY_datasets(config.data_path, config,  train=False)

    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')

    model_cfg = config.model_config
    if config.network == 'igmambanet':
        model = IGMambaNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
        )
        model.load_from()

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 512, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)

    step = 0
    print('#----------Training----------#')

    CUDA_LAUNCH_BLOCKING = 1

    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()

            preds.append(out)
            for i, predict in enumerate(preds):
                predict = np.where(predict >= 0.5, 1, 0)
                predict = predict.squeeze(0)
                predict = (predict * 255).astype(np.uint8)
    
                cv2.imwrite(os.path.join('{}/'.format(config.work_dir + 'outputs/') + str(i).zfill(4) + ".png"),
                            predict)

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
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0

        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, precision: {precision} confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

if __name__ == '__main__':
    config = setting_config
    main(config)