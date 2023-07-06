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


def choose_samples(data, choose_index):
    for key in data.keys():
        if key == 'ID' or key == 'Igg_V1' or key == 'Rn_interval' or key == 'Igg_V2' or key == 'Igg_V3' or key == 'Igg_V4' or key == 'Report9' or key == 'Report10' \
            or key == 'CT_feats' or key == 'CT_mask' or key == 'CT_interval' or key == 'Rn_interval'or key == 'CT_time':
            data[key]=data[key][choose_index]
        elif key == 'V1_time' or key == 'V2_time' or key == 'V3_time' or key == 'V4_time' or key == 'NAB_V1' or key == 'NAB_V2' or key == 'NAB_V3':
            data[key]=data[key][choose_index]
        elif key == 'raw_CTs':
            pass
        else:
            for subkey in data[key].keys():
                data[key][subkey]=data[key][subkey][choose_index]

def choose_samples_clinical(data, choose_index):
    for key in data.keys():
        if key == 'ID' or key == 'Igg_V1' or key == 'Rn_interval' or key == 'Igg_V2' or key == 'Igg_V3' or key == 'Igg_V4' or key == 'Report9' or key == 'Report10' \
            or key == 'CT_feats' or key == 'CT_mask' or key == 'CT_interval' or key == 'Rn_interval'or key == 'CT_time' or key == 'NAB_V1' or key == 'NAB_V2' or key == 'NAB_V3':
            data[key]=data[key][choose_index]
        elif key == 'V1_time' or key == 'V2_time' or key == 'V3_time' or key == 'V4_time' :
            data[key] = [data[key][28], data[key][96], data[key][144]]
        elif key == 'raw_CTs':
            pass
        else:
            for subkey in data[key].keys():
                data[key][subkey]=data[key][subkey][choose_index]

def choose_samples_CT(data, choose_index):
    for key in data.keys():
        data[key]=data[key][choose_index]

def val_data_for_specific_IgG(val_batch, igg_type):
    if igg_type=='Igg_V2':
        igg_time = 'V2_time'
    elif igg_type=='Igg_V3':
        igg_time = 'V3_time'
    elif igg_type=='Igg_V4':
        igg_time = 'V4_time'
    for j in range(len(val_batch['ID'])):
        for key in val_batch.keys():
            if key == 'Report1' or key == 'Report3' or key == 'Report4' or key == 'Report6' or key == 'Report7':

                val_batch[key]['interval'][j] = val_batch[key]['interval'][j]-val_batch[igg_time][j]
            val_batch['CT_interval'][j] = val_batch['CT_interval'][j] - val_batch[igg_time][j]


def val(model, val_data, config, epoch, data_name, num_fold, writer, Igg_pattern):
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
                choose_index = val_batch_copy[igg_type]>0
                choose_samples(val_batch_copy, choose_index.cpu())
                if igg_type!='Igg_V1':
                    val_data_for_specific_IgG(val_batch_copy, igg_type)
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
        precision = classification_report(labels.numpy(), scores.argmax(1), target_names=target_names, digits=3,output_dict=True)['weighted avg']['precision']
        recall = classification_report(labels.numpy(), scores.argmax(1), target_names=target_names, digits=3,output_dict=True)['weighted avg']['recall']
        f1 = classification_report(labels.numpy(), scores.argmax(1), target_names=target_names, digits=3,output_dict=True)['weighted avg']['f1-score']
        acc = classification_report(labels.numpy(), scores.argmax(1), target_names=target_names, digits=3,output_dict=True)['accuracy']
        
        print('Val confusion matrix: ')
        print(cm_all)
        print('Val precision: %.3f' % precision)
        print('Val recall: %.3f' % recall)
        print('Val f1: %.3f' % f1)
        print('Val acc: %.3f' % acc)

        model_state_dict = deepcopy(model.state_dict())
        model.train()
    return model_state_dict, precision, recall, f1, acc


def train(model, train_data, CT_features, config, num_fold, global_step, Igg_pattern):

    optimizer = optim.SGD(params=model.parameters(), lr=config.lr, momentum=0.9)
    if config.loss_type=='FocalLoss':
        criterion = MultiFocalLoss(config.num_classes, alpha=config.focal_alpha, gamma=config.focal_gamma)
    elif config.loss_type=='CE':
        criterion = nn.CrossEntropyLoss()

    train_dataloader = DataLoader(dataset_SD(train_data, config, CT_features=CT_features, raw_CT_path=config.raw_CT_path), batch_size=config.batch_size, shuffle=True,drop_last=False)
    val_dataloader = DataLoader(dataset_SD(val_data, config, CT_features=CT_features, raw_CT_path=config.raw_CT_path), batch_size=100, shuffle=False,drop_last=False)
    
    best_model_state_dict = model.state_dict()
    best_acc = -1
    ret_dict = {'aucs': -1, 'preds':None, 'labels':None}
    iters = len(train_dataloader)

    for epoch in range(config.max_epoch):
        comatrix_list=[]
        errors=[]
        gt_all=[]
        pred_all=[]
        i=0
        for data_src in tqdm(train_dataloader):
            
            if config.CTs_augmentation==True:
                data_src = CTs_augmentation_batch(config, data_src)

            optimizer.zero_grad()

            # randomly select a certain IgG in report12, Igg_V1, Igg_V2, Igg_V3, and Igg_V4 as the ground truth, then mask the data after the sampling time of the ground truth
            batch_size = data_src['Report12']['mask'].shape[0]
            IgG_mask = torch.cat((data_src['Report12']['mask'][:,0,:],
                       torch.zeros(batch_size,1).cuda(),
                       (1-(data_src['Igg_V2']>0).int()).unsqueeze(1),
                       (1-(data_src['Igg_V3']>0).int()).unsqueeze(1),
                       (1-(data_src['Igg_V4']>0).int()).unsqueeze(1)),1)

            IgGs = torch.cat((data_src['Report12']['raw_data'][:,1,:],data_src['Igg_V1'].unsqueeze(1),data_src['Igg_V2'].unsqueeze(1),data_src['Igg_V3'].unsqueeze(1),data_src['Igg_V4'].unsqueeze(1)),1)
            raw_gt = torch.zeros(batch_size)
            for j in range(batch_size):
                
                IgG_index = random.choices([0,1,2,3,4,5,6],weights=(1-IgG_mask[j]))[0]
                raw_gt[j] = IgGs[j][IgG_index]
                if IgG_index < 3:
                    for key in data_src.keys():
                        if key == 'Report1' or key == 'Report3' or key == 'Report4' or key == 'Report6' or key == 'Report7':
                            data_src[key]['mask'][j][data_src[key]['interval'][j] <= data_src['Report12']['interval'][j,1,IgG_index]] = 1
                            data_src[key]['interval'][j] = data_src[key]['interval'][j]-data_src['Report12']['interval'][j,1,IgG_index]

                    data_src['CT_mask'][j][data_src['CT_interval'][j] <= data_src['Report12']['interval'][j,1,IgG_index]] = 0
                    data_src['CT_interval'][j] = data_src['CT_interval'][j] - data_src['Report12']['interval'][j,1,IgG_index]

                elif IgG_index > 3:
                    if IgG_index==4:
                        for key in data_src.keys():
                            if key == 'Report1' or key == 'Report3' or key == 'Report4' or key == 'Report6' or key == 'Report7':
                                data_src[key]['interval'][j] = data_src[key]['interval'][j]-data_src['V2_time'][j]
                        data_src['CT_interval'][j] = data_src['CT_interval'][j] - data_src['V2_time'][j]
                    elif IgG_index==5:
                        for key in data_src.keys():
                            if key == 'Report1' or key == 'Report3' or key == 'Report4' or key == 'Report6' or key == 'Report7':
                                data_src[key]['interval'][j] = data_src[key]['interval'][j]-data_src['V3_time'][j]
                        data_src['CT_interval'][j] = data_src['CT_interval'][j] - data_src['V3_time'][j]
                    elif IgG_index==6:
                        for key in data_src.keys():
                            if key == 'Report1' or key == 'Report3' or key == 'Report4' or key == 'Report6' or key == 'Report7':
                                data_src[key]['interval'][j] = data_src[key]['interval'][j]-data_src['V4_time'][j]
                        data_src['CT_interval'][j] = data_src['CT_interval'][j] - data_src['V4_time'][j]
     
            pred = model(data_src)
            
            if Igg_pattern==True:
                
                bins_gt=torch.ones(data_src[config.IgG_version].shape).cuda()
                bins_gt[(data_src['Igg_V2']<config.IgG_threshold[0])*(data_src['Igg_V3']<config.IgG_threshold[0])]=0
                bins_gt[(data_src['Igg_V2']>=config.IgG_threshold[0])*(data_src['Igg_V3']>=config.IgG_threshold[0])]=2

            else:
                bins_gt=deepcopy(raw_gt.cuda())
                if config.num_classes==2:
                    bins_gt[data_src[config.IgG_version]>config.IgG_threshold[0]]=1
                    bins_gt[data_src[config.IgG_version]<=config.IgG_threshold[0]]=0
                elif config.num_classes==3:
                    bins_gt[data_src[config.IgG_version]>=config.IgG_threshold[1]]=2
                    bins_gt[(data_src[config.IgG_version]>=config.IgG_threshold[0])*(data_src[config.IgG_version]<config.IgG_threshold[1])]=1
                    bins_gt[data_src[config.IgG_version]<config.IgG_threshold[0]]=0
                elif config.num_classes==4:
                    bins_gt[data_src[config.IgG_version]>=config.IgG_threshold[2]]=3
                    bins_gt[(data_src[config.IgG_version]>=config.IgG_threshold[1])*(data_src[config.IgG_version]<config.IgG_threshold[2])]=2
                    bins_gt[(data_src[config.IgG_version]>=config.IgG_threshold[0])*(data_src[config.IgG_version]<config.IgG_threshold[1])]=1
                    bins_gt[data_src[config.IgG_version]<config.IgG_threshold[0]]=0

            confusion_matrix = meter.ConfusionMeter(config.num_classes)

            # metrics
            confusion_matrix.add(pred.detach().cpu(), bins_gt.cpu())
            comatrix_list.append(confusion_matrix.value())
            
            if i ==0:
                scores=pred.detach().cpu()
                labels=bins_gt.cpu()
            else:
                scores=torch.cat((scores,pred.detach().cpu()),axis=0)
                labels=torch.cat((labels,bins_gt.cpu()),axis=0)
            i+=1

            
            if config.loss_type=='LabelSmooth':
                loss = loss_label_smoothing(pred, bins_gt.long())
            else:
                loss = criterion(pred, bins_gt.long())
            loss.backward()
            optimizer.step()
            global_step += 1

        lr = optimizer.param_groups[0]['lr']

        print('Epoch ', epoch+1)
        print('current lr: %8.5f' % lr)
        print('Training loss: ', loss.item())

        print('Test on training data ...')
        cm_all = np.array(comatrix_list).sum(axis=0)
        print('Train confusion matrix: ')
        print(cm_all)

        if config.num_classes==2:
            auc_epoch = roc_auc_score(labels.numpy(), F.softmax(scores).numpy()[:,1])
        elif config.num_classes>2:
            auc_epoch = roc_auc_score(labels.numpy(), F.softmax(scores).numpy(), multi_class='ovr')
        
        print('Training AUC: ', auc_epoch)
        writer.add_scalar('Train/AUC', auc_epoch, global_step)
        writer.add_scalar('Loss', loss.item(), global_step)
        writer.add_scalar('LR', lr, global_step)
        
        if epoch == (config.max_epoch-1):
            print('=== START testing for fold %d ===' % fold_num)
            model_state_dict, val_precision, val_recall, val_f1, val_acc = val(model, val_dataloader, config, epoch, 'val', num_fold, writer, Igg_pattern)
            
    writer.close()
    model_ret = {'pre':val_precision,'recall':val_recall,'f1':val_f1,'acc':val_acc}
    return model_state_dict, model_ret

if __name__ == "__main__":
    config = edict()

    # path setting
    config.model_save_path = '/home/xxxx/works/COVID-19_LTAP/checkpoints/'
    data_root = '/home/xxxx/works/COVID-19_LTAP/data/'
    static_file = data_root + 'your_static_data.pkl'
    CT_file = data_root + 'your_CT_data.pkl'
    config.pretrained_model_path = None
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
    config.max_epoch = 1000
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
    config.embed_dim=128 # 192*2
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

    check_dir = time.strftime('%Y-%m-%d %H:%M:%S')
    os.mkdir(os.path.join('checkpoints', check_dir))

    fps = open(clinical_file, 'rb')
    src_train_data = pickle.load(fps)

    accs_a = np.zeros(config.repeat)
    sens_a = np.zeros(config.repeat)
    specs_a = np.zeros(config.repeat)
    df_acc_all = []
    train_start_time = time.time()

    # ==== split train\test data，compute mean and std ====
    # five fold cross-validation
    # split by age
    # initialize the five fold cross-validation fold division data normalization, otherwise the specified age division interval is invalid
    kf = RepeatedKFold_class_wise_split_by_age(n_splits=5, n_repeats=1, random_state=0, age_split=config.age_boundary)
    
    if config.IgG_pattern == True:
        choosed_index = (src_train_data['Igg_V1']>0)*(src_train_data['Igg_V2']>0)*(src_train_data['Igg_V3']>0)*(src_train_data['Igg_V4']>0)
    elif config.IgG_pattern == 'Neutralizing':
        choosed_index = (src_train_data['NAB_V1']>0)
    else:
        choosed_index = (src_train_data[config.IgG_version]>0)

    for key in src_train_data.keys():
        if key == 'ID' or key == 'Igg_V1' or key == 'Rn_interval' or key == 'Igg_V2' or key == 'Igg_V3' or key == 'Igg_V4' or key == 'NAB_V1' or key == 'NAB_V2' or key == 'NAB_V3':
            src_train_data[key]=src_train_data[key][choosed_index]
        elif key == 'V1_time' or key == 'V2_time' or key == 'V3_time' or key == 'V4_time':
            src_train_data[key]=np.array(src_train_data[key])[choosed_index]
        else:
            for subkey in src_train_data[key].keys():
                src_train_data[key][subkey]=src_train_data[key][subkey][choosed_index]

    bins_label = np.zeros(src_train_data[config.IgG_version].shape)
    if len(config.IgG_threshold)==1:
        bins_label[(src_train_data[config.IgG_version]>config.IgG_threshold)]=1
    elif len(config.IgG_threshold)==2:
        bins_label[(src_train_data[config.IgG_version]>=config.IgG_threshold[1])]=2
        bins_label[(src_train_data[config.IgG_version]<config.IgG_threshold[1])*(src_train_data[config.IgG_version]>=config.IgG_threshold[0])]=1
        bins_label[(src_train_data[config.IgG_version]<config.IgG_threshold[0])]=0
    
    kf_split = kf.split(src_train_data['Sdata']['data'][:,0], bins_label)
   
    model_rets = []
    for fold_num, (idx_train, idx_val) in enumerate(kf_split):
        if fold_num in config.available_fold:
            config.split_idx.append(idx_val)
            print('=== START training for fold %d ===' % fold_num)
            print('train sample number: ', len(idx_train))
            print('val sample number: ', len(idx_val))

            train_data = {'ID':-1, 'Sdata': -1, 'Report0': -1, 'Report1': -1, 'Report2': -1, 'Report3': -1, \
                'Report4': -1, 'Report5': -1, 'Report6': -1, \
                'Report7': -1, 'Report8': -1, 'Report11': -1, 'Report12': -1, 'Igg_V1':-1, 'Igg_V2':-1, 'Igg_V3':-1, 'Igg_V4':-1, 'Rn_interval':-1, 'V1_time':-1, 'V2_time':-1, 'V3_time':-1, 'V4_time':-1, 'NAB_V1':-1, 'NAB_V2':-1, 'NAB_V3':-1}
            val_data = {'ID':-1, 'Sdata': -1, 'Report0': -1, 'Report1': -1, 'Report2': -1, 'Report3': -1, \
                'Report4': -1, 'Report5': -1, 'Report6': -1, \
                'Report7': -1, 'Report8': -1, 'Report11': -1, 'Report12': -1, 'Igg_V1':-1, 'Igg_V2':-1, 'Igg_V3':-1, 'Igg_V4':-1, 'Rn_interval':-1, 'V1_time':-1, 'V2_time':-1, 'V3_time':-1, 'V4_time':-1, 'NAB_V1':-1, 'NAB_V2':-1, 'NAB_V3':-1}
            for key in train_data.keys():
                if (key!='Igg_V1' and key!='Igg_V2' and key!='Igg_V3' and key!='Igg_V4' and key!='ID' and key!='Rn_interval' and key!='V1_time' and key!='V2_time' and key!='V3_time' and key!='V4_time' and key!='NAB_V1' and key!='NAB_V2' and key!='NAB_V3'):
                    train_data[key] = {key:item[idx_train] for key, item in src_train_data[key].items()}
                    val_data[key] = {key:item[idx_val] for key, item in src_train_data[key].items()}
                elif (key=='ID' or key=='Igg_V1' or key=='Igg_V2' or key=='Igg_V3' or key=='Igg_V4' or key=='Rn_interval' or key=='V1_time' or key=='V2_time' or key=='V3_time' or key=='V4_time' or key=='NAB_V1' or key=='NAB_V2' or key=='NAB_V3'):
                    if key=='V1_time' or key=='V2_time' or key=='V3_time' or key=='V4_time':
                        train_data[key] = np.array(src_train_data[key])[idx_train]
                        val_data[key] = np.array(src_train_data[key])[idx_val]
                    else:
                        train_data[key] = src_train_data[key][idx_train]
                        val_data[key] = src_train_data[key][idx_val]

            use_cuda = True
            FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

            # Read CT features
            df=open(CT_file,'rb')
            CT_features = pickle.load(df)
            df.close()

            for data in [train_data,val_data]:
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

            if config.pretrained_model_path != None:
                weight = torch.load(config.pretrained_model_path)
                vit.load_state_dict(weight)

            model = MedicNet(config, vit)

            if torch.cuda.is_available():
                model.cuda()
                
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
            
            writer = SummaryWriter(comment=f'CTMo_{config.encoder_3d}_CLMo_{config.clinical_backbone}_F{fold_num}_LR{config.lr}_BS{config.batch_size}_EP{config.max_epoch}_Loss_{config.loss_type}_CT_Seq{config.seq_len}D{config.CT_feat_dim}CTfeat_{config.CT_data}')
            global_step = 0

            model_state_dict, model_ret = train(model, train_data, CT_features, config, fold_num, global_step, config.IgG_pattern)
            model_rets.append(model_ret)
            Model_name = 'Model_Fold'+str(config.available_fold[fold_num])+'_LR'+str(config.lr)+'_BS'+str(config.batch_size)+'_EP'+str(config.max_epoch)+'_CT_Seq'+str(config.seq_len)+'D'+str(config.CT_feat_dim)
            torch.save(model_state_dict, config.model_save_path+'/'+check_dir+'/'+Model_name+'.pkl')

    print('Training time: ', datetime.timedelta(seconds=(time.time()-train_start_time)))
    precision = 0
    recall = 0
    f1 = 0
    acc = 0

    folds_all_num = len(model_rets)
    for i in range(folds_all_num):
        precision += model_rets[i]['pre']
        recall += model_rets[i]['recall']
        f1 += model_rets[i]['f1']
        acc += model_rets[i]['acc']

    print('mean val precision: %.3f' % (precision/folds_all_num))
    print('mean val recall: %.3f' % (recall/folds_all_num))
    print('mean val f1: %.3f' % (f1/folds_all_num))
    print('mean val acc: %.3f' % (acc/folds_all_num))
