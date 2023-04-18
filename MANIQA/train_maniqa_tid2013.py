import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.maniqa import MANIQA
from config import Config
from utils.process import RandCrop, ToTensor, RandHorizontalFlip, Normalize, five_point_crop
from scipy.stats import spearmanr, pearsonr
from data.pipal21 import PIPAL21
from data.tid2013 import TID2013_train, TID2013
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

import argparse

from sklearn.model_selection import train_test_split, KFold



os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda()
        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()  
    
        pred_d = net(x_d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d)

            pred /= config.num_avg_val
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p


if __name__ == '__main__':
    cpu_num = 4
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


    parser = argparse.ArgumentParser(description="Parses for train and inf",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("seed", help="Random Seed", type=int)
    args = parser.parse_args()
    parsed_config = vars(args)
    SEED = parsed_config['seed']    

    setup_seed(SEED)

    initial_bs = 8
    initial_lr = 1e-5
    # !!! you need to change scheduler's T_max too 
    # to account for more items being with the current lr
    new_bs = 8
    new_lr = initial_lr * (new_bs/initial_bs)
    # config file
    config = Config({
        # dataset path
        "db_name": "TID2013",
        # path to dstorted images
        "train_dis_path": "C:\Users\pqlet\pass\IQA\datasets\tid2013\distorted_images",
        #"val_dis_path": "../../datasets/NITRE2022/NTIRE2022_NR_Valid_Dis/",
        # path to MOS_with_names files
        "train_txt_file_name": r"C:\Users\pqlet\pass\IQA\datasets\tid2013\mos_with_names.txt",
        #"val_txt_file_name": "./MANIQA/data/pipal21_val.txt",

        # optimization
        "batch_size": new_bs,
        "learning_rate": new_lr,
        "weight_decay": 1e-5,
        # We don't need more
        # The authors trained for 1 epoch
        # The shceduler steps every batch - that's why you don't log it
        # That's why 5 runs with its' different seeds
        "n_epoch": 1,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 5,
        "crop_size": 224,
        "num_workers": 4,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.13,
        
        # load & save checkpoint
        "model_name": f"model_maniqa__tid__seed{SEED}",
        
        "output_path": "./output",
        "snap_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/maniqa/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    logging.info(f"Seed : {SEED}")


    """
    list of seeds - 5 train__test_splits 
    10 fold run through EVERY training split 
    (10 times train on train + predict test) = 50 times
    """
    df_tif = pd.read_csv(
        r"C:\Users\pqlet\pass\IQA\datasets\tid2013\mos_with_names.txt", 
        sep=' ', 
        names=['MOS', 'img_filename']
    )

    kfold10 = KFold(n_splits=10, shuffle=True, random_state=42, )
    # Define the seeds in range() 
    # i used for seeding train_test_split
    for i in range(5):
        train_idx, test_idx = train_test_split(
            list(range(df_tif.shape[0])),
            test_size=0.2,
            random_state=i,
        )
        
        for fold_train_idx, fold_val_idx in kfold10.split(train_idx):
            train_df = df_tif.iloc[fold_train_idx]
            train_dataset = TID2013_pd(
                df = train_df,
                dis_path = config.train_dis_path,
                transform = transforms.Compose(
                [
                    RandCrop(224),
                    Normalize(0.5, 0.5),
                    RandHorizontalFlip(),
                    ToTensor()
                ]
            ))
            
            # Val dataset is not separate here
            # Validation subset is derived from the whole set
            # Test dataset also 
            # SO KEEP IN MIND TO OMIT SCORES WHEN IT'S VAL AND TEST DATALOADERS
            val_df = df_tif.iloc[fold_val_idx]
            val_dataset = TID2013_pd(
                df = val_df,
                dis_path = config.train_dis_path,
                transform = transforms.Compose(
                [
                    RandCrop(224),
                    Normalize(0.5, 0.5),
                    RandHorizontalFlip(),
                    ToTensor()
                ]
            ))
            
            




    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    
    net = MANIQA(
        embed_dim=config.embed_dim,
        num_outputs=config.num_outputs,
        dim_mlp=config.dim_mlp,
        patch_size=config.patch_size,
        img_size=config.img_size,
        window_size=config.window_size,
        depths=config.depths,
        num_heads=config.num_heads,
        num_tab=config.num_tab,
        scale=config.scale
    )
    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)
    
    
    
    # TO DO
    EMPLOY THE GroupKFold split FOR TID2013 so that there are no original images (i.e. with dist) in different folds to not leak 
    
    
    
    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        
        
        loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)
        
        # Saving training loss and metrics
        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("Train_SRCC", rho_s, epoch)
        writer.add_scalar("Train_PLCC", rho_p, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            
            # Saving validation loss and metrics
            writer.add_scalar("Val_loss", loss_val, epoch)
            writer.add_scalar("Val_SRCC", rho_s, epoch)
            writer.add_scalar("Val_PLCC", rho_p, epoch)
                        
            logging.info('Eval done...')

            if rho_s > best_srocc or rho_p > best_plcc:
                best_srocc = rho_s
                best_plcc = rho_p
                # save weights
                model_name = "epoch{}".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save(net, model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
