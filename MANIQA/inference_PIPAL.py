import os
import torch
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize, five_point_crop, sort_file
from data.tid2013 import TID2013
from tqdm import tqdm
import compute_metrics
from data.pipal22_test import PIPAL22

# My additional imports
from utils.inference_process import random_crop

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


def eval_epoch(config, net, test_loader, output_filename):
    with torch.no_grad():
        net.eval()
        name_list = []
        pred_list = []
        with open(os.path.join(config.valid_path, output_filename), 'w') as f:
            
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                    
                    if config.is_rand_crop:
                        # random 224x224 crop - they do 20 crops during inference for 1 image
                        x_d = random_crop(d_img=x_d, config=config)
                    else:
                        # # get one of 5 crops of the image
                        # # I don't see authors using it in the paper
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
        
    return pred_list, name_list


if __name__ == '__main__':
    print("cuda available -",torch.cuda.is_available())

    cpu_num = 2
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "db_name": "PIPAL",
        "test_dis_path": r"C:\Users\MQTyor\ai_pc\Reserch_ai\IQA\datasets\NITRE2022\NTIRE2022_NR_Valid_Dis",
        #"test_dis_path": r"C:\Users\MQTyor\ai_pc\Reserch_ai\IQA\datasets\NITRE2022\NTIRE2022_NR_Testing_Dis",
        
        # optimization
        "batch_size": 8,
        "num_avg_val": 5, # number of crops to average to get a final score for an image
        "crop_size": 224,
        "is_rand_crop": False,

        # device
        "num_workers": 2,

        # load & save checkpoint
        
        #"valid": "./output/valid",
        #"valid_path": "./output/valid/inference_valid",
        #"model_path": "./output/models/model_maniqa/epoch1"
        
        
        "valid": "output/valid",
        "valid_path": "output/valid/inf_PIPAL_table3_5seeds_pipal_random20",
        "model_object": 
            [
                r"C:\Users\MQTyor\ai_pc\Reserch_ai\IQA\projects\MANIQA_experiments\output\models\off_pipal\ckpt_valid",            
            ]
            # saved model from the official github repo - needed for state_dict loading
        #"model_pth": None, # new model or state_dict
    })
    
    
    assert len(config.model_object) > 0, "Empty list of checkpoints"
    assert len(config.model_object) is not None, "None model"
    
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
        drop_last=False, # It was True
        shuffle=False
    )
    
    scores = []
    name_list = []
    
    nets_filepath_list = []
    if type(config.model_object) is list:
        nets_filepath_list = config.model_object
    else:
        nets_filepath_list = [config.model_object,]
            
    #elif config.model_pth is not None:
    #    if type.config.model_pth == collections.OrderedDict :
    #        net.module.load_state_dict(torch.load(config.model_pth))
   
    
    for net_filepath in nets_filepath_list:
        net_filename = net_filepath.split(os.sep)[-2]
        output_filename = f'output_{net_filename}.txt'
        
        net = torch.load(net_filepath)
        net = net.cuda()
        scores_fold , name_list = eval_epoch(config, net, test_loader, output_filename)
        scores.append(scores_fold)
        sort_file(os.path.join(config.valid_path, output_filename))
        # PIPAL val and test don't have MOSes 
        # compute_metrics.main(config, net_filename, output_filename)
    
    avg_scores = np.array(scores).mean(axis=0)
    avg_out_filename = 'output_avg.txt'
    with open(os.path.join(config.valid_path, avg_out_filename), 'w') as f:
        for i in range(len(name_list)):
            f.write(name_list[i] + ',' + str(avg_scores[i]) + '\n')
    sort_file(os.path.join(config.valid_path, avg_out_filename))
    