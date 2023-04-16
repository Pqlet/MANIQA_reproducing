import os
import torch
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize, five_point_crop, sort_file
from data.pipal22_test import PIPAL22
from tqdm import tqdm


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


def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        name_list = []
        pred_list = []
        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    pred += net(x_d)

                pred /= config.num_avg_val
                d_name = data['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)
            for i in range(len(name_list)):
                f.write(name_list[i] + ',' + str(pred_list[i]) + '\n')
            print(len(name_list))
        f.close()


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


    parser = argparse.ArgumentParser(description="Parses for train and inf",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("seed", help="Random Seed")
    args = parser.parse_args()
    parsed_config = vars(args)
    SEED = parsed_config['seed']
    
    setup_seed(SEED)

    # config file
    config = Config({
        # dataset path
        "db_name": "PIPAL",
        "test_dis_path": r"C:\Users\MQTyor\ai_pc\Reserch_ai\IQA\datasets\NITRE2022\NTIRE2022_NR_Valid_Dis",
        #"test_dis_path": r"C:\Users\MQTyor\ai_pc\Reserch_ai\IQA\datasets\NITRE2022\NTIRE2022_NR_Testing_Dis",
        
        # optimization
        "batch_size": 10,
        # splits here
        "num_avg_val": 5, 
        "crop_size": 224,

        # device
        "num_workers": 8,

        # load & save checkpoint
        "valid": "./output/valid",
        "valid_path": "./output/valid/inf_pipal22_val__",
        "model_path": "./output/models/off_pipal/ckpt_valid"
    })

    if not os.path.exists(config.valid):
        os.mkdir(config.valid)

    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)
    
    # data load
    test_dataset = PIPAL22(
        dis_path=config.test_dis_path,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )
    net = torch.load(config.model_path)
    net = net.cuda()

    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    sort_file(config.valid_path + '/output.txt')
    