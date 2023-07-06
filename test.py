import pickle5 as pickle
import numpy as np
import os
import torch
from model import MedicNet, MultiFocalLoss
from copy import deepcopy
from torchnet import meter
import torch.optim as optim
import torch.nn as nn
import math
import time
from scipy import stats
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn.functional as F
from tqdm import tqdm
import time
import datetime
import random
from torch.utils.data.sampler import WeightedRandomSampler
from warmup_scheduler import GradualWarmupScheduler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, recall_score
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, MultimodalTransformer
from utils import dataset_SD, CTs_augmentation_batch, RepeatedKFold_class_wise_split_by_age, loss_label_smoothing


def val(model, val_data, config):
    with torch.no_grad():
        model.eval()
        comatrix_list=[]
        errors=[]
        gt_all=[]
        pred_all=[]
        for val_batch in tqdm(val_data):
            i=0
            for igg_type in config.val_IgG:
                val_batch_copy=deepcopy(val_batch)
                pred = model(val_batch_copy)

                bins_gt=deepcopy(val_batch_copy[igg_type])
                bins_gt[val_batch_copy[igg_type]>=config.IgG_threshold[1]]=2
                bins_gt[(val_batch_copy[igg_type]>=config.IgG_threshold[0])*(val_batch_copy[igg_type]<config.IgG_threshold[1])]=1
                bins_gt[val_batch_copy[igg_type]<config.IgG_threshold[0]]=0
                pred = model(val_batch_copy)

                confusion_matrix = meter.ConfusionMeter(config.num_classes)
                confusion_matrix.add(pred.detach().cpu(), bins_gt.cpu())
                comatrix_list.append(confusion_matrix.value())
                if i ==0:
                    scores=pred.detach().cpu()
                    labels=bins_gt.cpu()
                else:
                    scores=torch.cat((scores,pred.detach().cpu()),axis=0)
                    labels=torch.cat((labels,bins_gt.cpu()),axis=0)
                i+=1

        target_names = ['low', 'moderate', 'high']
        cm_all = np.array(comatrix_list).sum(axis=0)

        print('Val confusion matrix: ')
        print(cm_all)
        for i in range(len(val_batch_copy['ID'])):
            print(' The IgG level of patient ID %d is %s.' %(val_batch_copy['ID'][i], target_names[scores[i].argmax()]))

    
if __name__ == "__main__":
    config = edict()

    # path setting
    data_root = '/home/xxxx/works/COVID-19_LTAP/data/'
    static_file = data_root + 'samples_clinical.pkl'
    CT_file = data_root + 'samples_CT.pkl'
    config.pretrained_model_path = '/home/xxxx/works/COVID-19/checkpoints/pre-trained_LTAP_model.pkl'
    config.raw_CT_path = None

    # training setting
    config.IgG_threshold = [67,119]
    config.IgG_version = 'Igg_V1' # Igg_V1 / Igg_V2 / Igg_V3 / Igg_V4
    config.val_IgG = ['Igg_V1']
    config.IgG_pattern = 'False'
    config.available_fold = [0,1,2,3,4] # five folds
    config.gpu = '0'
    config.n_warmup_steps = 20
    config.lr_mul = 0.5
    config.lr = 5e-3
    config.weight_decay = 0.01
    config.grad_clip = 2
    config.fc_bias = False
    config.clinical_encoder_channels = [64,128]
    config.clinica_feat_dim = 394
    config.clinica_seq_len = [3,3,2,2,1,1,1,1,2,1] # Report1,Report2,Report3,Report4,Report5,Report6,Report7,Report8,Report11,Report12
    config.clinica_lstm_indim = config.clinical_encoder_channels[1]*2
    config.clinica_lstm_hidden_dim = config.clinica_lstm_indim*2
    config.CT_feat_dim = 512
    config.CT_feat_type = 'a' # type a: median+percentile+volume+percentage / type b: mean+max+volume+percentage
    config.lstm_indim = config.clinical_encoder_channels[1]*2
    config.hidden_dim = config.lstm_indim*2
    config.num_classes = len(config.IgG_threshold)+1
    config.seq_len = 11 # CT序列最大数据
    config.CT_seq_processor = 'lstm' # gru / lstm
    selected_feats = [0,7,9,22,23,31,37,48,49,50,55,56]
    config.mlp_indim = 383
    config.mlp_channels = [config.mlp_indim,config.mlp_indim,config.mlp_indim]
    config.batch_size = 400
    config.max_epoch = 10
    config.T_max = config.max_epoch
    config.lr_step_size = 100
    config.repeat = 1
    config.densenet_drop_rate = 0.5   
    config.mlp_drop_rate = 0.2
    config.encoder_3d = None # baseline / densenet / resnet / models_genesis / None
    config.sdata_pool = 'avg' # avg / max
    config.ddata_pool = 'avg' # avg / max
    config.init_tau = 5
    config.input_att = False # True/False
    config.clinical_backbone = 'transformer' # resnet / mlp+res / transformer
    config.lstm_all_output = True
    config.lstm_att = False
    config.loss_type = 'CE' # CE / LabelSmooth / FocalLoss
    config.input_dropout = False
    config.clinical_augmentation = False

    # dataset setting
    config.dataset_split_rate = 0.8
    config.clinical_type = 'bins' # original / bins
    config.CT_data = 'V6' 
    config.quantified_CT = None #  wPCA / woPCA / None
    config.focal_alpha = [1, 1, 1]
    config.focal_gamma = 2
    if config.IgG_version == 'Igg_V4':
        config.age_boundary = [-0.53, 0.4] # 77 patients
    else:
        config.age_boundary = [-0.4, 0.2] # 450 patients
    config.age_split_rate = 0.8
    config.split_type = 'split-by-age-cross-centers' 

    # normalization / augmentation setting
    config.normalize = 'Z-Score_indicator-wise'
    config.normalize_concated = False
    config.CTs_augmentation = False
    config.crop_len = 64
    config.CTs_crop = False 
    config.UDA = False
    config.split_idx=[]
    config.slice_number = 4

    # transformer
    config.transformer_type = 'vision' # multimodal / vision
    config.stop_grad_conv1 = False # True / False
    config.cuda = True
    config.ViT_pool = 'mean'  # 'mean' / 'cls'
    config.img_size=224
    config.patch_size=4
    config.in_chans=3
    config.vit_num_classes=10
    config.embed_dim=128
    config.depth=12
    config.num_heads=4
    config.mlp_ratio=2.
    config.qkv_bias=True
    config.representation_size=None # 768 / None
    config.distilled=False
    config.drop_rate=0.
    config.attn_drop_rate=0.
    config.drop_path_rate=0.
    config.mask_drop_rate=0.1
    config.embed_layer=PatchEmbed
    config.norm_layer=None
    config.act_layer=None
    config.weight_init=''

    # feature dimension and sequence length initialization
    config.d_sdata = 23
    config.d_report0 = 6
    config.n_position_R0 = 1
    config.d_report1 = 6
    config.n_position_R1 = 9
    config.d_report2 = 16
    config.n_position_R2 = 9
    config.d_report3 = 4
    config.n_position_R3 = 5
    config.d_report4 = 6
    config.n_position_R4 = 5
    config.d_report5 = 4
    config.n_position_R5 = 5
    config.d_report6 = 6
    config.n_position_R6 = 4
    config.d_report7 = 13
    config.n_position_R7 = 2
    config.d_report8 = 9
    config.n_position_R8 = 4

    # attention setting
    config.SA_reduction = 8
    config.GA_reduction = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    fps = open(static_file, 'rb')
    src_val_data = pickle.load(fps)

    accs_a = np.zeros(config.repeat)
    sens_a = np.zeros(config.repeat)
    specs_a = np.zeros(config.repeat)
    df_acc_all = []
    
    if config.IgG_pattern == True:
        choosed_index = (src_val_data['Igg_V1']>0)*(src_val_data['Igg_V2']>0)*(src_val_data['Igg_V3']>0)*(src_val_data['Igg_V4']>0)
    elif config.IgG_pattern == 'Neutralizing':
        choosed_index = (src_val_data['NAB_V1']>0)
    else:
        choosed_index = (src_val_data[config.IgG_version]>0)

    for key in src_val_data.keys():
        if key == 'ID' or key == 'Igg_V1' or key == 'Rn_interval' or key == 'Igg_V2' or key == 'Igg_V3' or key == 'Igg_V4' or key == 'NAB_V1' or key == 'NAB_V2' or key == 'NAB_V3':
            src_val_data[key]=src_val_data[key][choosed_index]
        elif key == 'V1_time' or key == 'V2_time' or key == 'V3_time' or key == 'V4_time':
            src_val_data[key]=np.array(src_val_data[key])[choosed_index]
        else:
            for subkey in src_val_data[key].keys():
                src_val_data[key][subkey]=src_val_data[key][subkey][choosed_index]

    bins_label = np.zeros(src_val_data[config.IgG_version].shape)
    if len(config.IgG_threshold)==1:
        bins_label[(src_val_data[config.IgG_version]>config.IgG_threshold)]=1
    elif len(config.IgG_threshold)==2:
        bins_label[(src_val_data[config.IgG_version]>=config.IgG_threshold[1])]=2
        bins_label[(src_val_data[config.IgG_version]<config.IgG_threshold[1])*(src_val_data[config.IgG_version]>=config.IgG_threshold[0])]=1
        bins_label[(src_val_data[config.IgG_version]<config.IgG_threshold[0])]=0
    
    use_cuda = True
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    # Read CT features
    df=open(CT_file,'rb')
    CT_features = pickle.load(df)
    df.close()

    data = src_val_data

    for key in data.keys():
        if key!='ID' and key!='Igg_V1' and key!='Igg_V2' and key!='Igg_V3' and key!='Igg_V4' and key!='V1_time' and key!='V2_time' and key!='V3_time' and key!='V4_time' and key!='Rn_interval' and key!='NAB_V1' and key!='NAB_V2' and key!='NAB_V3':
            data[key]['data'] = torch.from_numpy(data[key]['data']).type(FloatTensor)
            data[key]['data_bins'] = torch.from_numpy(data[key]['data_bins']).type(FloatTensor)
            data[key]['mask'] = torch.from_numpy(data[key]['mask']).type(FloatTensor)

            # Report internal invalid data filled with zero
            data[key]['data_bins'] = data[key]['data_bins']*(1-data[key]['mask'])
            data[key]['mean'] = torch.from_numpy(data[key]['mean']).type(FloatTensor)
            data[key]['std'] = torch.from_numpy(data[key]['std']).type(FloatTensor)
            data[key]['median'] = torch.from_numpy(data[key]['median']).type(FloatTensor)
                
            if key!='Sdata':
                data[key]['data_cls'] = torch.from_numpy(data[key]['data_cls']).type(FloatTensor)

            if key=='Report12':
                data[key]['raw_data'] = torch.from_numpy(data[key]['raw_data']).type(FloatTensor)

            data[key]['interval'] = torch.from_numpy(data[key]['interval']).type(FloatTensor)
            data[key]['interval_mask'] = torch.from_numpy(data[key]['interval_mask']).type(FloatTensor)
            data[key]['interval'] = data[key]['interval']*(1-data[key]['interval_mask'])

        elif key=='Igg_V1' or key=='Igg_V2' or key=='Igg_V3' or key=='Igg_V4' or key=='NAB_V1' or key=='NAB_V2' or key=='NAB_V3':
            data[key] = torch.from_numpy(data[key]).type(FloatTensor)

    CT_features['CT_feats'] = torch.from_numpy(CT_features['CT_feats']).type(FloatTensor)
    CT_features['CT_mask'] = torch.from_numpy(CT_features['CT_mask']).type(FloatTensor)

    # CT features Z-Score normalization
    CT_feats_mean = CT_features['CT_feats'].mean()
    CT_feats_std = CT_features['CT_feats'].std()
    CT_features['CT_feats'] = (CT_features['CT_feats']-CT_feats_mean)/CT_feats_std

    if config.transformer_type == 'vision':
        vit = VisionTransformer(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_chans=config.in_chans,
                num_classes=config.vit_num_classes,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                representation_size=config.representation_size,
                distilled=config.distilled,
                drop_rate=config.drop_rate,
                attn_drop_rate=config.attn_drop_rate,
                drop_path_rate=config.drop_path_rate,
                embed_layer=config.embed_layer,
                norm_layer=config.norm_layer,
                act_layer=config.act_layer,
                weight_init=config.weight_init,
                cls_token_enable=False,
                pool=config.ViT_pool,
                )
    elif config.transformer_type == 'multimodal':
        vit = MultimodalTransformer(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_chans=config.in_chans,
                num_classes=config.vit_num_classes,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                representation_size=config.representation_size,
                distilled=config.distilled,
                drop_rate=config.drop_rate,
                attn_drop_rate=config.attn_drop_rate,
                drop_path_rate=config.drop_path_rate,
                embed_layer=config.embed_layer,
                norm_layer=config.norm_layer,
                act_layer=config.act_layer,
                weight_init=config.weight_init,
                cls_token_enable=False,
                pool=config.ViT_pool,
                )

    model = MedicNet(config, vit)

    if config.pretrained_model_path != None:
        new_weight = OrderedDict()
        weight = torch.load(config.pretrained_model_path)
        for k, v in weight.items():
            name = k
            if k == 'Sdata_pos_embed.pos_table':
                continue
            if k == 'R1_pos_embed.pos_table':
                continue
            if k == 'R3_pos_embed.pos_table':
                continue
            if k == 'R4_pos_embed.pos_table':
                continue
            if k == 'R6_pos_embed.pos_table':
                continue
            if k == 'R7_pos_embed.pos_table':
                continue
            if k == 'CT_pos_embed.pos_table':
                continue
            new_weight[name] = v 
        model.load_state_dict(new_weight)

    if torch.cuda.is_available():
        model.cuda()
        
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    val_dataloader = DataLoader(dataset_SD(data, config, CT_features=CT_features, raw_CT_path=config.raw_CT_path), batch_size=3, shuffle=False,drop_last=False)
    val(model, val_dataloader, config)
    
