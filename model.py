import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchvision import models
from torchvision import transforms as tfs
from torch.autograd import Function
import random
import imageio


class IntervalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=20):
        super(IntervalEncoding, self).__init__()
        self.d_hid = d_hid
        self.n_position = n_position

    def _get_sinusoid_encoding_table(self, n_position, d_hid, interval):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(interval):

            interval[interval==0] = 200 
            return [interval.cpu().numpy()*40/200 / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array(get_position_angle_vec(interval))
        sinusoid_table = np.transpose(sinusoid_table,(1,2,0))
        sinusoid_table[:, :, 0::2] = np.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = np.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)

    def forward(self,x,interval):

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(self.n_position, self.d_hid, interval))

        return self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def plot_scan(scan,i):
    temp = '{}.png'.format(i+1)
    imageio.imwrite(os.path.join('./temp/', temp), scan)

def plot_CT_scans(data):
    scans_len = data.shape[0]
    for i in range(scans_len):
        plot_scan(data[i],i)

def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data - min)/(max-min+1e-6)

drop_rate = 0.2
drop_rate2 = 0.5 
NUM_CLASSES = 2
CHANNEL = 1

class Compute_class_score(nn.Module):
    def __init__(self,tau=3):
        super(Compute_class_score, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(tau)

    def forward(self, class0_avg_vector, class1_avg_vector, data_vector):

        class_scores = torch.zeros(2, data_vector.shape[0]).cuda()
        class0_simi = torch.exp(self.w1*torch.cosine_similarity(data_vector, class0_avg_vector, dim=1))
        class1_simi = torch.exp(self.w1*torch.cosine_similarity(data_vector, class1_avg_vector, dim=1))
        class_scores[0] = torch.div(class0_simi, (class0_simi+class1_simi))
        class_scores[1] = 1.0-class_scores[0]
        return class_scores.permute(1,0), float(self.w1)


class nonlinear_layer(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, in_dim, out_dim):
        super(nonlinear_layer, self).__init__()

        layers = []
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class project_attention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0):
        super(project_attention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.Fa_image = nonlinear_layer(image_feat_dim, hidden_size)
        self.Fa_txt = nonlinear_layer(txt_rnn_embeding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):        
        _, num_location, _ = image_feat.shape
        image_fa = self.Fa_image(image_feat)
        question_fa = self.Fa_txt(question_embedding)
        question_fa_expand = torch.unsqueeze(question_fa, 1).expand(-1, num_location, -1)
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        raw_attention = self.lc(joint_feature)
        return raw_attention

    def forward(self, image_feat, question_embedding):
        raw_attention = self.compute_raw_att(image_feat, question_embedding)
        # softmax across locations
        attention = F.softmax(raw_attention, dim=1).expand_as(image_feat)
        return attention


class self_attention(nn.Module):
    def __init__(self, channels, reduction):
        super(self_attention, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        module_input = x
        x = self.fc1(x.unsqueeze(2))
        x = self.relu(x)
        x = self.fc2(x)*np.sqrt(x.shape[1])
        x = self.softmax(x)
        x = module_input * x.squeeze() * x.shape[1]
        return x


class Baseline(nn.Module):

    def __init__(self, num_classes=2):
        super(Baseline, self).__init__()  
        
         #16x16X16X80 => 16x16x16x64
        self.conv1 = nn.Sequential(
            nn.Conv3d(CHANNEL, 64, kernel_size=3, padding=1),
            nn.ReLU())
        
        #16x16x16x64 => 8x8x8x64
        self.conv15 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1,stride=2),
            nn.ReLU())
        
        #8x8x8x64 => 8x8x8x128
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #8x8x8x128 => 4x4x4x128
        self.conv25 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #4x4x4x128 => 4x4x4x256
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #4x4x4x256 => 2x2x2x256
        self.conv35 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1,stride = 2),
            nn.BatchNorm3d(256),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #2x2x2x256 => 2x2x2x512
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #2x2x2x512 => 1x1x1x512
        self.conv45 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm3d(512),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #1x1x1x512
        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1,stride=1),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p =drop_rate2),
            nn.ReLU())
       
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, NUM_CLASSES, kernel_size=1,stride=1),
            nn.BatchNorm3d(NUM_CLASSES),
            nn.ReLU()) 
        
        self._convolutions = nn.Sequential(
            self.conv1,
            self.conv15,
            self.conv2,
            self.conv25,
            self.conv3,
            self.conv35,
            self.conv4,
            self.conv45,            
            self.conv5)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        out=self.conv1(x)
        out=self.conv15(out)
        out=self.conv2(out)
        out=self.conv25(out)
        out=self.conv3(out)
        out=self.conv35(out)
        out=self.conv4(out)
        out=self.conv45(out)            
        out=self.conv5(out)
        out = self.avg_pool(out)
        return out

# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model,n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        # final_out = self.base_model.out512
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        final_out = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        return final_out

class MedicNet(nn.Module):
    def __init__(self, config, vit):
        
        super(MedicNet, self).__init__()
        self.bias           = config.fc_bias
        self.mlp_indim      = config.mlp_indim
        self.mlp_middim     = self.mlp_indim
        self.lstm_indim     = config.lstm_indim
        self.hidden_dim     = config.hidden_dim
        self.num_classes    = config.num_classes
        self.seq_len        = config.seq_len
        self.slice_number   = config.slice_number
        self.batch_size     = config.batch_size
        self.mlp_drop_rate = config.mlp_drop_rate
        self.encoder_3d     = config.encoder_3d
        self.sdata_pool = config.sdata_pool
        self.input_att = config.input_att
        self.CT_feat_dim = config.CT_feat_dim
        self.clinical_backbone = config.clinical_backbone
        self.lstm_all_output = config.lstm_all_output
        self.lstm_att = config.lstm_att
        self.input_dropout = config.input_dropout
        self.clinical_augmentation = config.clinical_augmentation
        self.clinica_lstm_indim = config.clinica_lstm_indim
        self.clinica_lstm_hidden_dim = config.clinica_lstm_hidden_dim
        self.encoder_channels = config.clinical_encoder_channels
        self.clinical_encoder = vit
        self.stop_grad_conv1 = config.stop_grad_conv1

        device = torch.device('cuda' if config.cuda else 'cpu')

        self.d_sdata = config.d_sdata
        self.d_report0 = config.d_report0
        self.d_report1 = config.d_report1
        self.d_report2 = config.d_report2
        self.d_report3 = config.d_report3
        self.d_report4 = config.d_report4
        self.d_report5 = config.d_report5
        self.d_report6 = config.d_report6
        self.d_report7 = config.d_report7
        self.d_report8 = config.d_report8

        self.n_position_R0 = config.n_position_R0
        self.n_position_R1 = config.n_position_R1
        self.n_position_R2 = config.n_position_R2
        self.n_position_R3 = config.n_position_R3
        self.n_position_R4 = config.n_position_R4
        self.n_position_R5 = config.n_position_R5
        self.n_position_R6 = config.n_position_R6
        self.n_position_R7 = config.n_position_R7
        self.n_position_R8 = config.n_position_R8

        self.sdata_embedding = nn.Sequential(
            nn.Linear(config.d_sdata+config.d_report0, config.embed_dim),
            nn.BatchNorm1d(config.embed_dim),
            )

        self.report1_embedding = nn.Sequential(
            nn.Linear(config.d_report1, config.embed_dim),
            nn.BatchNorm1d(config.embed_dim),
            )
        self.report2_embedding = nn.Sequential(
            nn.Linear(config.d_report2, config.embed_dim),
            nn.BatchNorm1d(config.embed_dim),
            )
        self.report3_embedding = nn.Sequential(
            nn.Linear(config.d_report3, config.embed_dim),
            nn.BatchNorm1d(config.embed_dim),
            )
        self.report4_embedding = nn.Sequential(
            nn.Linear(config.d_report4, config.embed_dim),
            nn.BatchNorm1d(config.embed_dim),
            )
        self.report6_embedding = nn.Sequential(
            nn.Linear(config.d_report6, config.embed_dim),
            nn.BatchNorm1d(config.embed_dim),
            )
        self.report7_embedding = nn.Sequential(
            nn.Linear(config.d_report7, config.embed_dim),
            nn.BatchNorm1d(config.embed_dim),
            )
        
        self.Sdata_SA = self_attention(config.embed_dim,config.SA_reduction)
        self.R0_SA = self_attention(config.embed_dim,config.SA_reduction)
        self.R1_SA = self_attention(config.embed_dim,config.SA_reduction)
        self.R3_SA = self_attention(config.embed_dim,config.SA_reduction)
        self.R4_SA = self_attention(config.embed_dim,config.SA_reduction)
        self.R6_SA = self_attention(config.embed_dim,config.SA_reduction)
        self.R7_SA = self_attention(config.embed_dim,config.SA_reduction)
        self.CT_SA = self_attention(config.embed_dim,config.SA_reduction)
        
        self.R0_GA = project_attention(config.embed_dim,config.embed_dim,config.embed_dim*config.GA_reduction)
        self.R1_GA = project_attention(config.embed_dim,config.embed_dim,config.embed_dim*config.GA_reduction)
        self.R3_GA = project_attention(config.embed_dim,config.embed_dim,config.embed_dim*config.GA_reduction)
        self.R4_GA = project_attention(config.embed_dim,config.embed_dim,config.embed_dim*config.GA_reduction)
        self.R6_GA = project_attention(config.embed_dim,config.embed_dim,config.embed_dim*config.GA_reduction)
        self.R7_GA = project_attention(config.embed_dim,config.embed_dim,config.embed_dim*config.GA_reduction)
        self.CT_GA = project_attention(config.embed_dim,config.embed_dim,config.embed_dim*config.GA_reduction)

        self.Sdata_pos_embed = IntervalEncoding(config.embed_dim, 1)
        self.R0_pos_embed = IntervalEncoding(config.embed_dim, 1)
        self.R1_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R1)
        self.R2_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R2)
        self.R3_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R3)
        self.R4_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R4)
        self.R5_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R5)
        self.R6_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R6)
        self.R7_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R7)
        self.R8_pos_embed = IntervalEncoding(config.embed_dim, config.n_position_R8)
        self.CT_pos_embed = IntervalEncoding(config.embed_dim, config.seq_len)
        
        self.CT_feat_dim = config.CT_feat_dim
        self.CT_feat_type = config.CT_feat_type
        self.seq_len = config.seq_len

        self.CT_embedding = nn.Sequential(
            nn.Linear(config.CT_feat_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            )

        self.dim = config.embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.dropout = nn.Dropout(config.drop_rate)
        self.mask_dropout = nn.Dropout(config.mask_drop_rate)
        self.mask_drop_rate = config.mask_drop_rate

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm = nn.LayerNorm(self.dim)
        self.pool = config.ViT_pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim, int(self.dim)),
            nn.LayerNorm(int(self.dim)),
            nn.Linear(int(self.dim), self.num_classes),
        )

        if self.stop_grad_conv1:
            self.sdata_embedding[0].weight.requires_grad = False
            self.sdata_embedding[0].bias.requires_grad = False
            self.report0_embedding[0].weight.requires_grad = False
            self.report0_embedding[0].bias.requires_grad = False
            self.report1_embedding[0].weight.requires_grad = False
            self.report1_embedding[0].bias.requires_grad = False
            self.report3_embedding[0].weight.requires_grad = False
            self.report3_embedding[0].bias.requires_grad = False
            self.report4_embedding[0].weight.requires_grad = False
            self.report4_embedding[0].bias.requires_grad = False
            self.report6_embedding[0].weight.requires_grad = False
            self.report6_embedding[0].bias.requires_grad = False
            self.report7_embedding[0].weight.requires_grad = False
            self.report7_embedding[0].bias.requires_grad = False
            self.CT_embedding[0].bias.requires_grad = False

        if self.input_att == True:
            self.weights_sdata = torch.ones((1,config.d_sdata+config.d_report0)).cuda() * 0.5
            self.weights_r1 = torch.ones((1,config.d_report1)).cuda() * 0.5
            self.weights_r3 = torch.ones((1,config.d_report3)).cuda() * 0.5
            self.weights_r4 = torch.ones((1,config.d_report4)).cuda() * 0.5
            self.weights_r6 = torch.ones((1,config.d_report6)).cuda() * 0.5
            self.weights_r7 = torch.ones((1,config.d_report7)).cuda() * 0.5
            self.weights_ct = torch.ones((1,self.CT_feat_dim)).cuda() * 0.5

            self.att_sdata = nn.Sequential(
                            nn.Linear(config.d_sdata+config.d_report0, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, config.d_sdata+config.d_report0))

            self.att_r1 = nn.Sequential(
                            nn.Linear(config.d_report1, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, config.d_report1))

            self.att_r3 = nn.Sequential(
                            nn.Linear(config.d_report3, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, config.d_report3))

            self.att_r4 = nn.Sequential(
                            nn.Linear(config.d_report4, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, config.d_report4))

            self.att_r6 = nn.Sequential(
                            nn.Linear(config.d_report6, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, config.d_report6))

            self.att_r7 = nn.Sequential(
                            nn.Linear(config.d_report7, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, config.d_report7))

            self.att_ct = nn.Sequential(
                            nn.Linear(self.CT_feat_dim, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, self.CT_feat_dim))

        
    def forward(self, data, alpha = 1.0):
        batch_size=len(data['ID'])

        if self.CT_feat_dim == 768:
            ddata = data['CT_feats'][:,:self.seq_len,:self.CT_feat_dim]
        elif self.CT_feat_dim == 512:
            if self.CT_feat_type == 'a':
                ddata = torch.cat([data['CT_feats'][:,:self.seq_len,:256],data['CT_feats'][:,:self.seq_len,512:]],2)
            elif self.CT_feat_type == 'b':
                ddata = data['CT_feats'][:,:self.seq_len,256:]
        
        batch_size = data['Sdata']['data'].shape[0]

        sdata_input = torch.cat([data['Report0']['data'],data['Sdata']['data']],1)
        r1_input = data['Report1']['data'][:,:,:self.n_position_R1].permute(0,2,1).reshape(-1,self.d_report1)
        r3_input = data['Report3']['data'][:,:,:self.n_position_R3].permute(0,2,1).reshape(-1,self.d_report3)
        r4_input = data['Report4']['data'][:,:,:self.n_position_R4].permute(0,2,1).reshape(-1,self.d_report4)
        r6_input = data['Report6']['data'][:,:,:self.n_position_R6].permute(0,2,1).reshape(-1,self.d_report6)
        r7_input = data['Report7']['data'][:,:,:self.n_position_R7].permute(0,2,1).reshape(-1,self.d_report7)
        ct_input = ddata.reshape(-1,self.CT_feat_dim)

        if self.input_att>0:
            if self.training:
                # attention
                weights_sdata = self.att_sdata(sdata_input) 
                weights_sdata = weights_sdata.mean(dim=0).unsqueeze(0)
                weights_sdata = torch.sigmoid(weights_sdata)
                self.weights_sdata.detach()
                self.weights_sdata = self.weights_sdata * 0.95 + weights_sdata * 0.05

                weights_r1 = self.att_r1(r1_input) 
                weights_r1 = weights_r1.mean(dim=0).unsqueeze(0)
                weights_r1 = torch.sigmoid(weights_r1)
                self.weights_r1.detach()
                self.weights_r1 = self.weights_r1 * 0.95 + weights_r1 * 0.05

                weights_r3 = self.att_r3(r3_input) 
                weights_r3 = weights_r3.mean(dim=0).unsqueeze(0)
                weights_r3 = torch.sigmoid(weights_r3)
                self.weights_r3.detach()
                self.weights_r3 = self.weights_r3 * 0.95 + weights_r3 * 0.05

                weights_r4 = self.att_r4(r4_input) 
                weights_r4 = weights_r4.mean(dim=0).unsqueeze(0)
                weights_r4 = torch.sigmoid(weights_r4)
                self.weights_r4.detach()
                self.weights_r4 = self.weights_r4 * 0.95 + weights_r4 * 0.05

                weights_r6 = self.att_r6(r6_input) 
                weights_r6 = weights_r6.mean(dim=0).unsqueeze(0)
                weights_r6 = torch.sigmoid(weights_r6)
                self.weights_r6.detach()
                self.weights_r6 = self.weights_r6 * 0.95 + weights_r6 * 0.05

                weights_r7 = self.att_r7(r7_input) 
                weights_r7 = weights_r7.mean(dim=0).unsqueeze(0)
                weights_r7 = torch.sigmoid(weights_r7)
                self.weights_r7.detach()
                self.weights_r7 = self.weights_r7 * 0.95 + weights_r7 * 0.05

                weights_ct = self.att_ct(ct_input) 
                weights_ct = weights_ct.mean(dim=0).unsqueeze(0)
                weights_ct = torch.sigmoid(weights_ct)
                self.weights_ct.detach()
                self.weights_ct = self.weights_ct * 0.95 + weights_ct * 0.05

                if self.input_att==True:
                    sdata_input = weights_sdata * sdata_input
                    r1_input = weights_r1 * r1_input
                    r3_input = weights_r3 * r3_input
                    r4_input = weights_r4 * r4_input
                    r6_input = weights_r6 * r6_input
                    r7_input = weights_r7 * r7_input
                    ct_input = weights_ct * ct_input

            else:
                self.weights_sdata.detach()
                self.weights_r1.detach()
                self.weights_r3.detach()
                self.weights_r4.detach()
                self.weights_r6.detach()
                self.weights_r7.detach()
                self.weights_ct.detach()

                sdata_input = self.weights_sdata * sdata_input
                r1_input = self.weights_r1 * r1_input
                r3_input = self.weights_r3 * r3_input
                r4_input = self.weights_r4 * r4_input
                r6_input = self.weights_r6 * r6_input
                r7_input = self.weights_r7 * r7_input
                ct_input = self.weights_ct * ct_input

                print(self.weights_sdata)
                print(self.weights_r1)
                print(self.weights_r3)
                print(self.weights_r4)
                print(self.weights_r6)
                print(self.weights_r7)

        sdata_embed = self.sdata_embedding(sdata_input).unsqueeze(1)
        report1_embed = self.report1_embedding(r1_input).reshape(batch_size,-1,self.dim)
        report3_embed = self.report3_embedding(r3_input).reshape(batch_size,-1,self.dim)
        report4_embed = self.report4_embedding(r4_input).reshape(batch_size,-1,self.dim)
        report6_embed = self.report6_embedding(r6_input).reshape(batch_size,-1,self.dim)
        report7_embed = self.report7_embedding(r7_input).reshape(batch_size,-1,self.dim)
        CT_embed = self.CT_embedding(ct_input).reshape(batch_size,-1,self.dim)

        # self attention
        sdata_embed = self.Sdata_SA(sdata_embed.reshape(-1,self.dim)).reshape(batch_size,-1,self.dim)
        report1_embed = self.R1_SA(report1_embed.reshape(-1,self.dim)).reshape(batch_size,-1,self.dim)
        report3_embed = self.R3_SA(report3_embed.reshape(-1,self.dim)).reshape(batch_size,-1,self.dim)
        report4_embed = self.R4_SA(report4_embed.reshape(-1,self.dim)).reshape(batch_size,-1,self.dim)
        report6_embed = self.R6_SA(report6_embed.reshape(-1,self.dim)).reshape(batch_size,-1,self.dim)
        report7_embed = self.R7_SA(report7_embed.reshape(-1,self.dim)).reshape(batch_size,-1,self.dim)
        CT_embed = self.CT_SA(CT_embed.reshape(-1,self.dim)).reshape(batch_size,-1,self.dim)

        if self.training:
            sdata_mask = (self.mask_dropout(1-data['Sdata']['mask'][:,0])*(1-self.mask_drop_rate)).unsqueeze(1)
            report1_mask = self.mask_dropout(1-data['Report1']['mask'][:,0,:self.n_position_R1])*(1-self.mask_drop_rate)
            report3_mask = self.mask_dropout(1-data['Report3']['mask'][:,0,:self.n_position_R3])*(1-self.mask_drop_rate)
            report4_mask = self.mask_dropout(1-data['Report4']['mask'][:,0,:self.n_position_R4])*(1-self.mask_drop_rate)
            report6_mask = self.mask_dropout(1-data['Report6']['mask'][:,0,:self.n_position_R6])*(1-self.mask_drop_rate)
            report7_mask = self.mask_dropout(1-data['Report7']['mask'][:,0,:self.n_position_R7])*(1-self.mask_drop_rate)
            CT_mask = self.mask_dropout(data['CT_mask'][:,:self.seq_len])*(1-self.mask_drop_rate)
        else:
            sdata_mask = 1-data['Sdata']['mask'][:,0].unsqueeze(1)
            report1_mask = 1-data['Report1']['mask'][:,0,:self.n_position_R1]
            report3_mask = 1-data['Report3']['mask'][:,0,:self.n_position_R3]
            report4_mask = 1-data['Report4']['mask'][:,0,:self.n_position_R4]
            report6_mask = 1-data['Report6']['mask'][:,0,:self.n_position_R6]
            report7_mask = 1-data['Report7']['mask'][:,0,:self.n_position_R7]
            CT_mask = data['CT_mask'][:,:self.seq_len]

        Sdata_pos_embed = self.Sdata_pos_embed(sdata_embed,torch.cat([data['Sdata']['interval'][:,0].unsqueeze(1),data['Report0']['interval'][:,0].unsqueeze(1)],1))
        R1_pos_embed = self.R1_pos_embed(report1_embed,data['Report1']['interval'][:,0,:self.n_position_R1])
        R3_pos_embed = self.R3_pos_embed(report3_embed,data['Report3']['interval'][:,0,:self.n_position_R3])
        R4_pos_embed = self.R4_pos_embed(report4_embed,data['Report4']['interval'][:,0,:self.n_position_R4])
        R6_pos_embed = self.R6_pos_embed(report6_embed,data['Report6']['interval'][:,0,:self.n_position_R6])
        R7_pos_embed = self.R7_pos_embed(report7_embed,data['Report7']['interval'][:,0,:self.n_position_R7])
        CT_pos_embed = self.CT_pos_embed(CT_embed,data['CT_interval'][:,:self.seq_len])

        x_pos_embed = torch.cat([
            Sdata_pos_embed, 
            R1_pos_embed,
            R3_pos_embed,
            R4_pos_embed,
            R6_pos_embed,
            R7_pos_embed,
            CT_pos_embed,
            ], dim=1)

        x_mask = torch.cat([
            sdata_mask, 
            report1_mask,
            report3_mask,
            report4_mask,
            report6_mask,
            report7_mask,
            CT_mask,
            ], dim=1)

        x = torch.cat([
            sdata_embed, 
            report1_embed,
            report3_embed,
            report4_embed,
            report6_embed,
            report7_embed,
            CT_embed,
            ], dim=1)

        x = self.clinical_encoder(x, x_pos_embed.cuda(), x_mask)
        pred = self.mlp_head(x)
        return pred

    def save(self, check_path, name):
        if os.path.exists(check_path):
            os.mkdir(check_path)
        torch.save(self.state_dict(), os.path.join(check_path, name))
        print(os.path.join(check_path, name) + '\t saved!')

    def _make_layer(self,  inchannel, outchannel, infeat_dim, outfeat_dim, block_num, stride=1):
        '''
        build a layer that includes multiple residual blocks
        '''
        shortcut = nn.Sequential(
                nn.Conv1d(inchannel,outchannel,1,stride=2, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.ReLU(inplace=True))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, infeat_dim, outfeat_dim, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, outfeat_dim, outfeat_dim))
        return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    '''
    sub-module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, infeat_dim, outfeat_dim, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, 1, stride, 0, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv1d(outchannel, outchannel, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True))

        self.right = shortcut
        self.fc_left = nn.Linear(infeat_dim, outfeat_dim, bias=False)
        self.bn_left = nn.BatchNorm1d(outchannel)
        self.drop_left = nn.Dropout(p=0.5)
                
        self.fc_right = nn.Linear(infeat_dim, outfeat_dim, bias=False)
        self.bn_right = nn.BatchNorm1d(outchannel)
        self.infeat_dim = infeat_dim
        self.outfeat_dim = outfeat_dim

    def forward(self, x):

        # left branch
        out = self.left(x)
        size = out.size()
        out = self.fc_left(out.view(-1,self.infeat_dim))
        out = self.drop_left(out)
        out = out.view(size[0],size[1],self.outfeat_dim)

        # right branch
        residual = x if self.right is None else self.right(x)

        out += residual
        return torch.relu(out)

class MultiFocalLoss(nn.Module):
    """
    Reference : https://www.zhihu.com/question/367708982/answer/985944528
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = torch.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss