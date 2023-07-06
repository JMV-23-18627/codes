import pickle
import numpy as np
import os
import torch
from copy import deepcopy
from sklearn.model_selection import KFold, RepeatedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from sklearn import preprocessing
import random
import pdb
from tqdm import tqdm
import imageio
import datetime
import torch.nn.functional as F


def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss


def CTs_augmentation_batch(config, data):
    batch_shape=data['CTs'].shape
    augmented_batch=torch.zeros((batch_shape[0],batch_shape[1],batch_shape[2],config.crop_len,config.crop_len))

    for k in range(7):
        for j in range(data['CTs'].shape[0]):
            aug_mode_1=str(random.sample(range(0,3),1)[0])
            aug_mode_2=str(random.sample(range(0,4),1)[0])
            aug_mode = aug_mode_1+aug_mode_2
            augmented_batch[j,:,k,:,:]=torch.from_numpy(CTs_augmentation(data['CTs'][j,:,k,:,:].cpu().numpy(), aug_mode, crop=config.CTs_crop, crop_len=config.crop_len))

    data['CTs']=augmented_batch.cuda()

    return data


class RepeatedKFold_class_wise_split_by_age(RepeatedKFold):
    # stratified by age
    def __init__(self, n_splits=5, n_repeats=10, random_state=None, extra=['test', None], age_split=[]):
        super().__init__(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = n_splits
        self.age_split = age_split

    def age_stratified(self, X, y):
        """
        X is the column vector of age
        y is the category label
        """
        land = np.logical_and
        lor = np.logical_or
        left = -1000
        ret_mask = []
        for right in self.age_split:
            m = land(X > left, X < right+0.0001)
            ret_mask.append(m[:, None]) 
            left = right
        m = X > left
        ret_mask.append(m[:, None])
        mask = np.concatenate(ret_mask, axis=1)
        m = y == 1
        temp = deepcopy(mask)
        mask[m] = False
        temp[~m] = False
        mask = np.concatenate([mask, temp], axis=1)
        ind = np.arange(mask.shape[1])[None,].repeat(mask.shape[0], axis=0)
        stratified_ind = ind[mask]
        return stratified_ind
    
    def split(self, X, y):
        n_class = set(y).__len__()
        generators_idx = []
        index = []
        ind = self.age_stratified(X, y)
        ind_set = set(ind)
        for i in ind_set:
            index.append(np.arange(X.shape[0])[ind==i])
            X_i = X[ind==i]
            generators_idx.append(super().split(X_i))
            
        for i in range(self.n_repeats*self.n_splits):
            idx = [next(subidx) for subidx in generators_idx]
            idx_train = np.concatenate([index[i][subidx[0]] for i, subidx in enumerate(idx)])
            idx_test = np.concatenate([index[i][subidx[1]] for i, subidx in enumerate(idx)])

            yield idx_train, idx_test


class dataset_SD(Dataset):
    def __init__(self, data, config, CT_features=None, raw_CT_path=None):
        super().__init__()
        self.data = data
        self.CT_features = CT_features
        self.feats_mean = None if 'feats_mean' not in config.keys() else config.feats_mean
        self.feats_std = None if 'feats_std' not in config.keys() else config.feats_std
        self.raw_CT_path = raw_CT_path
    
    def set_CTs_mean_std(self, dic):
        self.CTs_mean = dic['CTs_mean']
        self.CTs_std = dic['CTs_std']
        print('Mean for all CT scans: ')
        print(self.CTs_mean)
        print('Std for all CT scans: ')
        print(self.CTs_std)
        
    def get_valid_mask(self):
        
        with open(self.mask_file, 'rb') as fp:
            valid = pickle.load(fp)
        return valid

        # hierarchical reading of clinical data
    def __getitem__(self, idx):
        ret = {'ID':-1, 'Sdata': -1, 'Report0': -1, 'Report1': -1, 'Report2': -1, 'Report3': -1, 'Report4': -1, 'Report6': -1, \
               'Report7': -1, 'Report8': -1, 'Report9': -1, 'Report10': -1, 'Report11': -1, 'Report12': -1, 'Igg_V1':-1, 'Igg_V2':-1, 'V2_time':-1, 'Igg_V3':-1, 'V3_time':-1, 'Igg_V4':-1, 'V4_time':-1, 'NAB_V1':-1, 'NAB_V2':-1, 'NAB_V3':-1, 'CT_feats':-1, \
               'CT_mask':-1, 'CT_interval':np.zeros((self.CT_features['CT_time'].shape[1])),'raw_CTs':[]}
        for key in self.data.keys():
            if key=='ID' or key=='Igg_V1' or key=='Igg_V2' or key=='Igg_V3' or key=='Igg_V4' or key=='V1_time' or key=='Rn_interval' or key=='IgG_pattern' or key=='NAB_V1' or key=='NAB_V2' or key=='NAB_V3':
                ret[key] = self.data[key][idx]
            elif key=='V2_time' or key=='V3_time' or key=='V4_time':
                if self.data[key][idx]!='NA':
                    ret[key] = (self.data['V1_time'][idx] - self.data[key][idx]).days
            else:
                ret[key] = {subkey: self.data[key][subkey][idx] for subkey in self.data[key].keys()}
            
        if self.CT_features != None:
            curr_index = self.CT_features['followup1_ID'].index(ret['ID'])
            ret['CT_feats'] = self.CT_features['CT_feats'][curr_index]
            ret['CT_mask'] = self.CT_features['CT_mask'][curr_index]
            ret['CT_time'] = self.CT_features['CT_time'][curr_index]
        
        for i in torch.where(ret['CT_mask']==1)[0].cpu().numpy().tolist():
            curr_CT_time_list = []
            curr_CT_time_list.append(int(str(ret['CT_time'][i])[:4]))
            curr_CT_time_list.append(int(str(ret['CT_time'][i])[4:6]))
            curr_CT_time_list.append(int(str(ret['CT_time'][i])[6:8]))
            curr_CT_time_list.append(int(str(ret['CT_time'][i])[8:10]))
            curr_CT_time_list.append(int(str(ret['CT_time'][i])[10:12]))
            curr_CT_time = datetime.datetime(*curr_CT_time_list)
            ret['CT_interval'][i] = (ret['V1_time'] - curr_CT_time).days

        if ret['ID'] == 1322:
            ret['CT_interval'][0]=0
            ret['CT_mask'][0]=0

        if self.raw_CT_path!=None:
            raw_CTs = []
            for i in range(10):
                if self.CT_features['CT_ID'][curr_index][i]==220021600139:
                    ret['CT_mask'][2]=0

                if self.CT_features['CT_mask'][curr_index][i]!=0 and (self.CT_features['CT_ID'][curr_index][i]!='220021600139'):
                    df = open(self.raw_CT_path+str(int(self.CT_features['CT_ID'][curr_index][i]))+'.pickle','rb')
                    curr_raw_CT = pickle.load(df)
                    raw_CTs.append(curr_raw_CT)
                else:
                    raw_CTs.append(-1)

            for i in range(len(raw_CTs)):
                if ret['CT_mask'][i]==1:
                    if i ==0:
                        ret['raw_CTs']=np.expand_dims(raw_CTs[i]['sub_volumes'],0)
                    else:
                        ret['raw_CTs']=np.concatenate((ret['raw_CTs'],np.expand_dims(raw_CTs[i]['sub_volumes'],0)),0)
                else:
                    if i ==0:
                        ret['raw_CTs']=np.zeros((1,16,64,64,32))
                    else:
                        ret['raw_CTs']=np.concatenate((ret['raw_CTs'],np.zeros((1,16,64,64,32))),0)

        ret.pop('V1_time')
        ret['CT_interval']=torch.from_numpy(ret['CT_interval']).cuda()

        return ret

    def load_CT_file(self, order): 
        """
        order is the number of the CT file (patient number)
        """    
        pickle_file = os.path.join(self.pickle_dir, '%d.pickle'%order)
        try:
            with open(pickle_file, 'rb') as fp:
                CTs = pickle.load(fp)
                
                CTs = CTs if order < 500 else CTs['3D_image']
        except FileNotFoundError:
            CTs = []
        return CTs
    
    def valid_CTs_mean_std(self):
        print('Start compute mean and std for all valid CT scans...')
        orders = self.data['order']
        means = []
        stds = []
        for idx in tqdm(range(orders.__len__())):
            data = self.getitem(idx)
            if data['ct_mask'].sum() < 1:
                continue
            imgs = data['CTs'][data['ct_mask']] 
            
            nonzero_index=imgs>0

            # calculate mean and std in the lung area
            means.append(imgs[nonzero_index].mean().unsqueeze(0))
            stds.append(imgs[nonzero_index].std().unsqueeze(0))

        mean = torch.cat(means, dim=0).mean(dim=0)
        std = torch.cat(stds, dim=0).mean(dim=0)
        self.CTs_mean = mean
        self.CTs_std = std
        print('Mean for all CT scans: ')
        print(mean)
        print('Std for all CT scans: ')
        print(std)
        return {'CTs_mean':mean, 'CTs_std':std}

    def getitem_normalize(self, idx): 
        """
        1. zero invalid CT
        2. normalize valid CT
        """
        data = self.getitem(idx)
        m = data['mask']
        nonzero_index=data['CTs'][m]>0
        data['CTs'][~m] = 0.
        data['CTs'][m] = (data['CTs'][m] - self.CTs_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / (self.CTs_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+1e-8)
            
        return data
    

    def img_tranform_3(self, img):
        
        center = -500 #window width and window position of the lung CT
        width = 1500
        # convert to window width and window level
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)
        img = img - min
        img = torch.trunc( img * dFactor)
        img[img < 0.0] = 0
        img[img > 255.0] = 255
        # convert to floating number of [0, 1]
        img /= 255.
        return img
    

    def img_tranform(self, img):
        img[img==-3000] = 0
        
        return img

    def CTs_normalization(image_3d, norm_type='z-score'):
        for i in range(image_3d.shape[0]):
            if norm_type=='z-score':
                image_3d[i] = z_score_3d(image_3d[i].squeeze()).unsqueeze(0)

            elif norm_type=='zero-one':
                image_3d[i] = zero_one_scale_3d(image_3d[i].squeeze()).unsqueeze(0)

        return image_3d


    def transform(self, vec):
        T = [mapping[trans](param_dict) for trans, param_dict in self.t_dict.items()]
        return Compose(T)(vec)

    def __len__(self):
        return self.data['Igg_V1'].__len__()

class dataset(Dataset):
    def __init__(self, data, config, only_sdata=True):
        super().__init__()
        self.data = data
        self.only_sdata = only_sdata
        self.feats_mean = None if 'feats_mean' not in config.keys() else config.feats_mean
        self.feats_std = None if 'feats_std' not in config.keys() else config.feats_std
        
    def set_CTs_mean_std(self, dic):
        self.CTs_mean = dic['CTs_mean']
        self.CTs_std = dic['CTs_std']
        print('Mean for all CT scans: ')
        print(self.CTs_mean)
        print('Std for all CT scans: ')
        print(self.CTs_std)
        
    def get_valid_mask(self):
        
        with open(self.mask_file, 'rb') as fp:
            valid = pickle.load(fp)
        return valid

    def __getitem__(self, idx):
        if self.feats_mean is None or self.feats_std is None:
            raise ValueError('Please compute features mean and std first BY \n 1. set set_feats_mean_std \n 2. valid_feats_mean_std')
   
        if self.only_sdata:
             ret = {key: self.data[key][idx] for key in self.data.keys()}
        else:
            ret = self.getitem_normalize(idx)
 
        return ret

    def load_CT_file(self, order): 
        """
        order is the number of the CT file (patient number)
        """    
        pickle_file = os.path.join(self.pickle_dir, '%d.pickle'%order)
        try:
            with open(pickle_file, 'rb') as fp:
                CTs = pickle.load(fp)
                
                CTs = CTs if order < 500 else CTs['3D_image']
        except FileNotFoundError:
            CTs = []
        return CTs
    
    def valid_CTs_mean_std(self):
        print('Start compute mean and std for all valid CT scans...')
        orders = self.data['order']
        means = []
        stds = []
        for idx in tqdm(range(orders.__len__())):
            data = self.getitem(idx)
            if data['ct_mask'].sum() < 1:
                continue
            imgs = data['CTs'][data['ct_mask']] 
            
            nonzero_index=imgs>0

            # calculate mean and std in the lung area
            means.append(imgs[nonzero_index].mean().unsqueeze(0))
            stds.append(imgs[nonzero_index].std().unsqueeze(0))

        mean = torch.cat(means, dim=0).mean(dim=0)
        std = torch.cat(stds, dim=0).mean(dim=0)
        self.CTs_mean = mean
        self.CTs_std = std
        print('Mean for all CT scans: ')
        print(mean)
        print('Std for all CT scans: ')
        print(std)
        return {'CTs_mean':mean, 'CTs_std':std}
    
    def getitem_normalize(self, idx): 
        """
        1. zero invalid CT
        2. normalize valid CT
        """
        data = self.getitem(idx)
        m = data['ct_mask']
        nonzero_index=data['CTs'][m]>0
        data['CTs'][~m] = 0.
        data['CTs'][m] = (data['CTs'][m] - self.CTs_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / (self.CTs_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+1e-8)
            
        return data
        
    def getitem(self, idx):        
        ret = {key: self.data[key][idx] for key in self.data.keys()}
        return ret

    def img_tranform_3(self, img):
        
        center = -500 # window width and window position of the lung CT
        width = 1500
        # convert to window width and window level
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)
        img = img - min
        img = torch.trunc( img * dFactor)
        img[img < 0.0] = 0
        img[img > 255.0] = 255
        # convert to floating number of [0, 1]
        img /= 255.
        return img
    

    def img_tranform(self, img):
        img[img==-3000] = 0
        
        return img

    def CTs_normalization(image_3d, norm_type='z-score'):
        for i in range(image_3d.shape[0]):
            if norm_type=='z-score':
                image_3d[i] = z_score_3d(image_3d[i].squeeze()).unsqueeze(0)

            elif norm_type=='zero-one':
                image_3d[i] = zero_one_scale_3d(image_3d[i].squeeze()).unsqueeze(0)

        return image_3d


    def transform(self, vec):
        T = [mapping[trans](param_dict) for trans, param_dict in self.t_dict.items()]
        return Compose(T)(vec)

    def __len__(self):
        return self.data['labels'].__len__()