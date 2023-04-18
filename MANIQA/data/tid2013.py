import os
import torch
import numpy as np
import cv2 



class TID2013_pd(torch.utils.data.Dataset):
    def __init__(self, df, dis_path, transform):
        super(TID2013_pd, self).__init__()
        self.df = df
        self.dis_path = dis_path
        self.transform = transform
        
        
        dis_files_data, score_data = [], []
        dis_files_data = self.df['img_filename'].values
        score_data = self.df['MOS'].values
    
        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        
        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class TID2013(torch.utils.data.Dataset):
    def __init__(self, dis_path, transform):
        super(TID2013, self).__init__()
        self.dis_path = dis_path
        self.transform = transform

        dis_files_data = []
        for dis in os.listdir(dis_path):
            dis_files_data.append(dis)
        self.data_dict = {'d_img_list': dis_files_data}

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        sample = {
            'd_img_org': d_img,
            'd_name': d_img_name
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

