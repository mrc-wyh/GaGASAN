import pandas as pd
import numpy as np
import dgl
import math
import pickle
import os, sys
os.chdir(sys.path[0])

class Data(object):
    def __init__(self, args):
        dataset = args.dataset
        
        loc_file = dataset + '/loc.csv'
        if args.dist == 1000:
            geo_edge_file = dataset + '/geo_edge.csv'
        elif args.dist == 500:
            geo_edge_file = dataset + '/geo_edge_500.csv'
        elif args.dist == 1500:
            geo_edge_file = dataset + '/geo_edge_1500.csv'
        else:
            geo_edge_file = dataset + '/geo_edge_2000.csv'
        tran_edge_file = dataset + '/tran_edge.csv'
        user_id_file = dataset + '/user_id.csv'
        train_forward_file = dataset + '/train_forward.pickle'
        train_labels_file = dataset + '/train_labels.pickle'
        train_user_file = dataset + '/train_user.pickle'
        valid_forward_file = dataset + '/valid_forward.pickle'
        valid_labels_file = dataset + '/valid_labels.pickle'
        valid_user_file = dataset + '/valid_user.pickle'
        test_forward_file = dataset + '/test_forward.pickle'
        test_lables_file = dataset + '/test_labels.pickle'
        test_user_file = dataset + '/test_user.pickle'
        
        loc = pd.read_csv(loc_file, names=['loc_ID', 'latitude', 'longitude', 'loc_new_ID'], sep=',', header=0)
        user = pd.read_csv(user_id_file, names=['old_id', 'new_id'], sep=',', header=0)
        geo_edge = pd.read_csv(geo_edge_file, names=['src', 'dst'], sep=',', header=0)
        tran_edge = pd.read_csv(tran_edge_file, names=['src', 'dst', 'freq', 'weight'], sep=',', header=0)
        self.loc_num = max(loc['loc_new_ID']) + 1
        self.user_num = max(user['new_id']) + 1
        self.loc_g, self.tran_edge_weight = self.build_graph(geo_edge, tran_edge)
        print(self.loc_num, self.user_num)
        
        train_forward = open(train_forward_file,'rb')
        self.train_forward = pickle.load(train_forward)
        train_labels = open(train_labels_file,'rb')
        self.train_labels = pickle.load(train_labels)
        train_user = open(train_user_file,'rb')
        self.train_user = pickle.load(train_user)
        valid_forward = open(valid_forward_file,'rb')
        self.valid_forward = pickle.load(valid_forward)
        valid_labels = open(valid_labels_file,'rb')
        self.valid_labels = pickle.load(valid_labels)
        valid_user = open(valid_user_file,'rb')
        self.valid_user = pickle.load(valid_user)
        test_forward = open(test_forward_file,'rb')
        self.test_forward = pickle.load(test_forward)
        test_labels = open(test_lables_file,'rb')
        self.test_labels = pickle.load(test_labels)
        test_user = open(test_user_file,'rb')
        self.test_user = pickle.load(test_user)
        print('train traj num:', len(self.train_forward))
        print('valid traj num:', len(self.valid_forward))
        print('test traj num:', len(self.test_forward))
      
        
    def build_graph(self, geo_edge, tran_edge):
        geo = np.array(geo_edge)
        geo_e = [tuple(geo[i]) for i in range(len(geo))]
        tran = np.array(tran_edge[['src', 'dst']])
        tran_e_w = np.array(tran_edge['weight'])
        tran_e = [tuple(tran[i]) for i in range(len(tran))]
        data_dict = {
            ('loc', 'geo', 'loc'): geo_e,
            ('loc', 'trans', 'loc'): tran_e
        }
        return dgl.heterograph(data_dict), tran_e_w
    
    def cal_freq_list(self, max_radius, min_radius, frequency_num):
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = (1.0/timescales).astype(np.float32)
        return freq_list
    
    def cal_freq_mat(self, freq_list):
        freq_mat = np.expand_dims(freq_list, axis = 1)
        freq_mat = np.repeat(freq_mat, 2, axis = 1)
        return freq_mat
    
    def make_input_embeds(self, coords, frequency_num, freq_mat):
        loc_num = coords.shape[0]
        coor_dim = coords.shape[1]
        coords_mat = coords
        coords_mat = np.expand_dims(coords_mat, axis = 2)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        coords_mat = np.repeat(coords_mat, frequency_num, axis = 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 3)
        spr_embeds = coords_mat * freq_mat

        spr_embeds[:, :, :, 0::2] = np.sin(spr_embeds[:, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, 1::2] = np.cos(spr_embeds[:, :, :, 1::2])  # dim 2i+1

        spr_embeds = np.reshape(spr_embeds, (loc_num, -1))
        return spr_embeds 