import torch
from torch.cuda.amp import GradScaler
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
warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    model_path = r'/media/share02/fcp/packs/pycharm_project_907/VM-UNet-main/results/vmunet_IEAhou_Monday_24_March_2025_16h_41m_27s'
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

    data_path = 'VM-UNet-main/data/isic2017/val/images'

    test_dataset = NPY_datasets(config.data_path, config,  train=False)

    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')

    model_cfg = config.model_config
    # if config.network == 'vmunet':
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            # load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 512, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    # criterion = config.criterion_val
    optimizer = get_optimizer(config, model)

    step = 0
    print('#----------Training----------#')

    CUDA_LAUNCH_BLOCKING = 1

    # model_Loader = os.path.join(checkpoint_dir, 'best-epoch268-loss0.1592.pth')
    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    # switch to evaluate mode
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
                #     # 将预测结果乘以 255 并转换为无符号 8 位整型
                predict = np.where(predict >= 0.5, 1, 0)
                predict = predict.squeeze(0)
                #
                predict = (predict * 255).astype(np.uint8)
                #
                cv2.imwrite(os.path.join('{}/'.format(config.work_dir + 'outputs/') + str(i).zfill(4) + ".png"),
                            predict)
            # for i, predict in enumerate(preds):
            #     # 将预测结果乘以 255 并转换为无符号 8 位整型
            #     predict = predict.squeeze(0)
            #
            #     predict = (predict * 255).astype(np.uint8)
            #     # 保存图像
            #     cv2.imwrite(os.path.join('{}/'.format(config.work_dir + 'outputs/') + str(i).zfill(4) + ".png"), predict)

            #save_imgs2(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=None)

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

        # if test_data_name is not None:
        #     log_info = f'test_datasets_name: {test_data_name}'
        #     print(log_info)
        #     logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, precision: {precision} confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


# def save_imgs2(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
#     # img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     # img = img / 255. if img.max() > 1.1 else img
#
#     output_dir = os.path.join(save_path, 'msk_preds')
#     os.makedirs(output_dir, exist_ok=True)
#     msk_pred = np.where(np.squeeze(msk_pred, axis=0) > 0.4, 255, 0)
#     # 将预测结果转换为图像格式
#     cv2.imwrite(os.path.join(output_dir, f'{str(i).zfill(6)}.png'), msk_pred)  # 保存图像


    # if datasets == 'retinal':
    #     msk = np.squeeze(msk, axis=0)
    #     msk_pred = np.squeeze(msk_pred, axis=0)
    # else:
    #     msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
    #     msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)




    #
    # plt.figure(figsize=(7, 15))
    #
    # plt.subplot(3, 1, 1)
    # plt.imshow(img)
    # plt.axis('off')
    #
    # plt.subplot(3, 1, 2)
    # plt.imshow(msk, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(3, 1, 3)
    # plt.imshow(msk_pred, cmap='gray')
    # plt.axis('off')
    #
    # if test_data_name is not None:
    #     save_path = save_path + test_data_name + '_'
    # plt.savefig(save_path + str(i) + '.png')
    # plt.close()

if __name__ == '__main__':
    config = setting_config
    main(config)