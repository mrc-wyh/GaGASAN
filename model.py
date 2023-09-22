import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os, sys
os.chdir(sys.path[0])


class GeoGCNLayer(nn.Module):
    def __init__(self,
                g,
                args,
                device='cuda'
                ):
        super(GeoGCNLayer, self).__init__()
        self.g = g
        self.device = device
        self.act = nn.LeakyReLU(0.2)
        self.is_att = args.is_att 
        self.is_sgc = args.is_sgc
        self.geo_w = args.geo_w
        self.tran_w = args.tran_w
        self.is_lightgcn = args.is_lightgcn
        if self.is_att:
            self.attn_fuse = SemanticAttention(args.hidden_dim, args.hidden_dim*4)
        if not args.is_lightgcn:
            self.feat_tran = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.feat_tran.weight, gain=1.414)
        
    def forward(self, feat):
        funcs = {}#message and reduce functions dict
        feat_t = feat if self.is_lightgcn else self.feat_tran(feat)
        self.g.ndata['f'] = feat_t
        for srctype, etype, dsttype in self.g.canonical_etypes:
            if etype == 'geo':
                if self.geo_w == 0 and not self.is_att:
                    continue
                else:
                    funcs[etype] = (fn.copy_u('f', 'm'), fn.mean('m', 'geo'))
            else:
                if self.tran_w == 0 and not self.is_att:
                    continue
                else:
                    funcs[etype] = (fn.u_mul_e('f', 'w', 'm'), fn.sum('m', 'trans'))
                    
        self.g.multi_update_all(funcs, 'sum')
        if self.is_att: 
            geo = self.g.ndata['geo'].unsqueeze(1)
            trans = self.g.ndata['trans'].unsqueeze(1)
            z = torch.cat([geo, trans], 1)
            feat = self.attn_fuse(z)
        else:
            if self.geo_w == 0:
                feat = self.g.ndata['trans']
            elif self.tran_w == 0:
                feat = self.g.ndata['geo']
            else:
                feat = self.geo_w * self.g.ndata['geo'] + self.tran_w * self.g.ndata['trans']
        return feat if self.is_sgc else self.act(feat)

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    
        beta = torch.softmax(w, dim=0)                
        beta = beta.expand((z.shape[0],) + beta.shape) 
        return (beta * z).sum(1)                       
                                              
class GeoGCN(nn.Module):
    def __init__(self,
                g,
                tran_e_w,
                args,
                device='cuda'
                ):
        super(GeoGCN, self).__init__()
        g = g.int()
        g = dgl.remove_self_loop(g, etype='geo')
        g = dgl.add_self_loop(g, etype='geo')
        self.g = g.to(device)
        self.g.edges['trans'].data['w'] = torch.tensor(tran_e_w).float().to(device)
        self.num_layer = args.GeoGCN_layer_num
        self.dropout = args.gcn_drop
        self.device = device
        self.act = nn.LeakyReLU(0.2)

        self.gcn = nn.ModuleList()
        for i in range(self.num_layer):
            self.gcn.append(
            GeoGCNLayer(self.g, args, device)
        )
            
    def forward(self, feat):
        for i in range(self.num_layer - 1):
            feat = self.gcn[i](feat)
        if self.num_layer > 1:
            feat = F.dropout(feat, self.dropout)
        feat = self.gcn[-1](feat)
        return feat

class SlotEncoding(nn.Module):
    "Position Encoding module"
    def __init__(self, dim_model, max_len=100, device='cuda'):
        super(SlotEncoding, self).__init__()
        pe = torch.zeros(max_len, dim_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict
    
    def forward(self, pos):
        return torch.index_select(self.pe, 1, pos).squeeze(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SeqPred(nn.Module):
    def __init__(self,
                loc_num,
                args,
                device='cuda'
                ):
        super(SeqPred, self).__init__()
        self.dim = args.hidden_dim
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(args.hidden_dim, args.enc_drop)
        encoder_layers = TransformerEncoderLayer(args.hidden_dim, args.enc_nhead, args.enc_ffn_hdim, args.enc_drop)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.enc_layer_num)
        self.decoder = nn.Linear(args.hidden_dim, loc_num)
        self.init_weights()
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        
    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        
        src = src * math.sqrt(self.dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask) #shape * batch *dim
        return self.decoder(output)

class FFN(nn.Module):
    '''
    This module implements feed-forward network(after the Multi-Head Network) equation:
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    '''
    def __init__(self, inpu_dim, hidden, out_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(inpu_dim, hidden)
        self.w_2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class DatasetPrePare(Dataset):
    def __init__(self, forward, label, user):
        self.forward = forward
        self.label = label
        self.user = user

    def __len__(self):
        assert len(self.forward) == len(self.label) == len(self.user)
        return len(self.forward)

    def __getitem__(self, index):
        return (self.forward[index], self.label[index], self.user[index])

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_epoch_val_loss = 0
        
    def step(self, score, loss, user_model, loc_model, gcn_model, enc_model, epoch, result_dir):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, loc_model, gcn_model, enc_model, result_dir)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = loss
            self.save_checkpoint(user_model, loc_model, gcn_model, enc_model, result_dir)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, user_model, loc_model, gcn_model, enc_model, result_dir):
        state_dict = {
            'user_emb_model_state_dict': user_model.state_dict(),
            'loc_emb_model_state_dict': loc_model.state_dict(),
            'geogcn_model_state_dict': gcn_model.state_dict(),
            'transformer_encoder_model_state_dict': enc_model.state_dict()
            } 
        best_result = os.path.join(result_dir, 'checkpoint.pt')    
        torch.save(state_dict, best_result)

'''
class Space2Vec(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000, min_radius = 10,
            freq_init = "geometric",
            ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(Space2Vec, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim 
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()
        self.ffn = ffn
        
    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis = 1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")
        
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat
        
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        
        spr_embeds = self.make_input_embeds(coords)

        # # loop over all batches

        # spr_embeds: shape (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds)

        # return sprenc
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds
'''        