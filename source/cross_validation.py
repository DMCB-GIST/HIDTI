import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import random as rn
import pandas as pd
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

from random import shuffle
from copy import deepcopy
from itertools import product
from arguments import argparser, logging
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.preprocessing import MinMaxScaler

from heterogeneous_data import *


np.random.seed(1)
rn.seed(1)



TABSY = "\t"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, fold_info=0, epoch_info=0, log_info='empty', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.fold_info = fold_info
        self.epoch_info = epoch_info
        self.log_info = log_info

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold_info, epoch_info, log_info):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_path = '/home/'+str(log_info)+'_'+str(fold_info)+'_'+str(epoch_info)+'_checkpoint.pt'
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

def loadDataset_dti(X, Y, Xall, Yall, labels, batch_size):
    x_len = len(X)
    y_len = len(Y)
    if(x_len<batch_size):
        data_all = np.concatenate((X, Xall, Y, Yall), 1)
        data_all = [data_all]

        data_label = [labels]
        one_count = labels.count(1)
        batch_len=1
    else:
        batch_len = math.floor(x_len/batch_size)

        if(x_len==y_len):
            data_all = []
            data_label = []
            one_count = []
            idx = 0
            for i in range(batch_len):
                tmp_idx = idx+batch_size
                tmp_x = X[idx:tmp_idx]
                tmp_xall = Xall[idx:tmp_idx]
                tmp_y = Y[idx:tmp_idx]
                tmp_yall = Yall[idx:tmp_idx]
                tmp = np.concatenate((tmp_x, tmp_xall, tmp_y, tmp_yall), axis=1)
                data_all.append(tmp)

                tmp_label = labels[idx:tmp_idx]
                tmp_count = tmp_label.count(1)
                one_count.extend([tmp_count])
                data_label.append(tmp_label)
                idx = idx+batch_size
        else:
            print("length error!")

    return data_all, data_label, batch_len, one_count

class dta_model(nn.Module):
    def __init__(self, NUM_FILTERS, FILTER_LENGTH2, input_dim):
        super(dta_model, self).__init__()
        self.relu = nn.ReLU()
        self.leakyReLU = torch.nn.LeakyReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(p=0.5)
        #dvec=300, ddi:707, dse:4192, ddis:5603 = 6711
        #pvec=100, ppi:1489, psim:1489, pdis:5603 = 8681

        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LayerNorm(int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2),int(input_dim/8)),
            )
        self.projection1 = torch.nn.Sequential(
            nn.Linear(input_dim,int(input_dim/8))
            )
        
        self.out = torch.nn.Sequential(
            nn.LayerNorm(int(input_dim/8)),
            nn.ReLU(),
            nn.Linear(int(input_dim/8), int(input_dim/32)),
            nn.LayerNorm(int(input_dim/32)),
            nn.ReLU(),
            nn.Linear(int(input_dim/32), 1),
            )

    def forward(self,dr1,dr3,dr4,pr1,pr2,pr3,pr4):
        concat = torch.cat([dr1, dr3, dr4, pr1, pr2, pr3, pr4], dim=1)

        x1 = self.layer1(concat)
        x1 += self.projection1(concat)

        predictions = self.out(x1)

        return torch.sigmoid(predictions)   

class EarlyStopping_ddi:
    def __init__(self, patience=15, verbose=False, fold_info=0, epoch_info=0, log_info='empty', delta=0):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.fold_info = fold_info
        self.epoch_info = epoch_info
        self.log_info = log_info

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold_info, epoch_info, log_info):
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_path = '/home/dis_'+str(log_info)+'_'+str(fold_info)+'_'+str(epoch_info)+'_checkpoint.pt'
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

def loadDataset_ddi(X, labels, batch_size):
    x_len = len(X)

    if(x_len<batch_size):
        data_one = [X]
        data_label = [labels]
        batch_len=1
    else:
        batch_len = math.floor(x_len/batch_size)

        data_one = []
        data_label = []
        idx = 0
        for i in range(batch_len):
            tmp_idx = idx+batch_size
            tmp_x = X[idx:tmp_idx]
            data_one.append(tmp_x)
                
            tmp_label = labels[idx:tmp_idx]
            data_label.append(tmp_label)
            idx = idx+batch_size
    
    return data_one, data_label, batch_len

class drugint_model(nn.Module):
    def __init__(self, DVEC_SIZE):
        super(drugint_model, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(DVEC_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 707)

        self.drop = nn.Dropout(p=0.1)

    def forward(self,x):
        encode_smiles = x

        FC1 = self.relu(self.fc1(x))
        FC1 = self.drop(FC1)
        FC2 = self.relu(self.fc2(FC1))
        FC2 = self.drop(FC2)
        FC3 = self.relu(self.fc3(FC2))
        FC3 = self.drop(FC3)
        predictions = self.out(FC3)

        return torch.sigmoid(predictions)

class EarlyStopping_dse:
    def __init__(self, patience=15, verbose=False, fold_info=0, epoch_info=0, log_info='empty', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.fold_info = fold_info
        self.epoch_info = epoch_info
        self.log_info = log_info

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold_info, epoch_info, log_info):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_path = '/home/dse_'+str(log_info)+'_'+str(fold_info)+'_'+str(epoch_info)+'_checkpoint.pt'
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

def loadDataset_dse(X, labels, batch_size):
    x_len = len(X)

    if(x_len<batch_size):
        data_one = [X]
        data_label = [labels]
        batch_len=1
    else:
        batch_len = math.floor(x_len/batch_size)

        data_one = []
        data_label = []
        idx = 0
        for i in range(batch_len):
            tmp_idx = idx+batch_size
            tmp_x = X[idx:tmp_idx]
            data_one.append(tmp_x)
                
            tmp_label = labels[idx:tmp_idx]
            data_label.append(tmp_label)
            idx = idx+batch_size
    
    return data_one, data_label, batch_len

class side_model(nn.Module):
    def __init__(self, DVEC_SIZE):
        super(side_model, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(DVEC_SIZE, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024,4192)

        self.drop = nn.Dropout(p=0.1)

    def forward(self,x):
        encode_smiles = x

        FC1 = self.relu(self.fc1(x))
        FC1 = self.drop(FC1)
        FC2 = self.relu(self.fc2(FC1))
        FC2 = self.drop(FC2)
        FC3 = self.relu(self.fc3(FC2))
        FC3 = self.drop(FC3)
        predictions = self.out(FC3)

        return torch.sigmoid(predictions)

class EarlyStopping_dis:
    def __init__(self, patience=15, verbose=False, fold_info=0, epoch_info=0, log_info='empty', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.fold_info = fold_info
        self.epoch_info = epoch_info
        self.log_info = log_info

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold_info, epoch_info, log_info):
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_path = '/home/dis_'+str(log_info)+'_'+str(fold_info)+'_'+str(epoch_info)+'_checkpoint.pt'
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

def loadDataset_dis(X, labels, batch_size):
    x_len = len(X)

    if(x_len<batch_size):
        data_one = [X]
        data_label = [labels]
        batch_len=1
    else:
        batch_len = math.floor(x_len/batch_size)

        data_one = []
        data_label = []
        idx = 0
        for i in range(batch_len):
            tmp_idx = idx+batch_size
            tmp_x = X[idx:tmp_idx]
            data_one.append(tmp_x)
                
            tmp_label = labels[idx:tmp_idx]
            data_label.append(tmp_label)
            idx = idx+batch_size

    return data_one, data_label, batch_len

class disease_model(nn.Module):
    def __init__(self, DVEC_SIZE):
        super(disease_model, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(DVEC_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1512)

        self.drop = nn.Dropout(p=0.1)

    def forward(self,x):
        encode_smiles = x

        FC1 = self.relu(self.fc1(x))
        FC1 = self.drop(FC1)
        FC2 = self.relu(self.fc2(FC1))
        FC2 = self.drop(FC2)
        FC3 = self.relu(self.fc3(FC2))
        FC3 = self.drop(FC3)
        predictions = self.out(FC3)

        return torch.sigmoid(predictions)

class EarlyStopping_pdis:
    def __init__(self, patience=15, verbose=False, fold_info=0, epoch_info=0, log_info='empty', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.fold_info = fold_info
        self.epoch_info = epoch_info
        self.log_info = log_info

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.fold_info, self.epoch_info, self.log_info)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold_info, epoch_info, log_info):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_path = '/home/pdis_'+str(log_info)+'_'+str(fold_info)+'_'+str(epoch_info)+'_checkpoint.pt'
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

def loadDataset_pdis(X, labels, batch_size):
    x_len = len(X)

    if(x_len<batch_size):
        data_one = [X]
        data_label = [labels]
        batch_len=1
    else:
        batch_len = math.floor(x_len/batch_size)

        data_one = []
        data_label = []
        idx = 0
        for i in range(batch_len):
            tmp_idx = idx+batch_size
            tmp_x = X[idx:tmp_idx]
            data_one.append(tmp_x)
                
            tmp_label = labels[idx:tmp_idx]
            data_label.append(tmp_label)
            idx = idx+batch_size
    
    return data_one, data_label, batch_len

class pdisease_model(nn.Module):
    def __init__(self, PVEC_SIZE):
        super(pdisease_model, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(PVEC_SIZE, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, 5603)
        self.drop = nn.Dropout(p=0.1)

    def forward(self,x):
        encode_smiles = x

        FC1 = self.relu(self.fc1(x))
        FC1 = self.drop(FC1)
        FC2 = self.relu(self.fc2(FC1))
        FC2 = self.drop(FC2)
        FC3 = self.relu(self.fc3(FC2))
        FC3 = self.drop(FC3)
        predictions = self.out(FC3)

        return torch.sigmoid(predictions)


def cross_validation_DTI(dvec, pvec, Y_DTI, ddivec, dsevec, disvec, pdisvec, label_row_inds, label_col_inds, train_index, valid_index, test_index, FLAGS)
    epoch = FLAGS.num_epoch
    batchsz = FLAGS.batch_size
    tmp = FLAGS.output_path
    tmp = tmp.split('/')
    tmp_log = tmp[1]


    all_predictions = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] 
    all_losses = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    ddi_test_aucs = []
    dse_test_aucs = []
    dis_test_aucs = []
    pdis_test_aucs = []
    for foldind in range(0,len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]
        testinds = test_sets[foldind]
       
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]
        train_drugs, train_Y = prepare_interaction_pairs_ddi(ddivec, Y, trrows, trcols)
       
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        val_drugs, val_Y = prepare_interaction_pairs_ddi(ddivec, Y, terows, tecols)

        testrows = label_row_inds[testinds]
        testcols = label_col_inds[testinds]
        test_drugs, test_Y = prepare_interaction_pairs_ddi(ddivec, Y, testrows, testcols)

        tmp_drugs = []
        for i in range(0, len(test_drugs)):
            test_drugs[i][300:] = 0
            tmp_drugs.append(list(test_drugs[i][:300]))
        tmp_drugs = np.stack(tmp_drugs)

        ddi_dat, ddi_lab, ddi_num = loadDataset_ddi(train_drugs, train_Y, batchsz)
        ddi_val_dat, ddi_val_lab, ddi_val_num = loadDataset_ddi(val_drugs, val_Y, batchsz)
        ddi_test_dat, ddi_test_lab, ddi_test_num = loadDataset_ddi(test_drugs, test_Y, batchsz)

        ddi_model = drugint_model(1007)
        ddi_model.to(device)
        criterion = nn.BCELoss()
        learning_rate = 1e-5
        optimizer = optim.Adam(ddi_model.parameters(), lr=learning_rate)

        auc_vals = 0.0
        ddi_epoch = 1000
        early_stopping_ddi = EarlyStopping_ddi(verbose=True, fold_info=foldind, epoch_info=ddi_epoch, log_info=tmp_log)
        for i in range(ddi_epoch):
            ddi_model.train()
            print('Epoch {}/{}'.format(i, ddi_epoch - 1))

            running_loss = 0.0
            num_preds_dti = 0
            running_loss_ddi = 0.0
            num_preds_ddi = 0

            running_loss = 0.0
            for j in range(0,ddi_num):
                x = ddi_dat[j]
                x = torch.from_numpy(x).float()
                x = Variable(x).to(device)

                label = ddi_lab[j]
                label = torch.FloatTensor(label)
                label = Variable(label).to(device)

                outputs = ddi_model(x)

                loss = criterion(outputs, label).to(device)
                for name, param in ddi_model.named_parameters():
                    if 'weight' in name:
                        L1_1 = Variable(param, requires_grad=True)
                        L1_2 = torch.norm(L1_1, 2)
                        L1_3 = 10e-4*L1_2
                        loss = loss + L1_3
                running_loss += loss.item()*x.size(0)

                num_preds_dti += x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss0 = running_loss/num_preds_dti

            val_preds = []
            val_labels = []

            val_preds_ddi = []
            val_labels_ddi = []

            ddi_model.eval()
            with torch.no_grad():
                for param in ddi_model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                val_running_loss = 0.0
                val_num_preds = 0
                val_running_loss_ddi = 0.0
                val_num_preds_ddi = 0
                for j in range(0,ddi_val_num):
                    x = ddi_val_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    val_label = ddi_val_lab[j]
                    new_label = [i for x in val_label for i in x]
                    val_labels.extend(new_label)
                    val_label = torch.FloatTensor(val_label)
                    val_label = Variable(val_label).to(device)

                    val_outputs = ddi_model(x)

                    val_loss = criterion(val_outputs, val_label).to(device)
                    val_running_loss += val_loss.item()*x.size(0)
                    val_num_preds += x.size(0)

                    val_outputs = val_outputs.view(-1).tolist()

                    val_preds.extend(val_outputs)

                val_epoch_loss = val_running_loss/val_num_preds
            auc_val = auc_score(val_labels, val_preds)
 
            print('DDI prediction training loss: {:.4f}, val_loss : {:.4f}, val_auc: {:.4f}'.format(epoch_loss0, val_epoch_loss, auc_val))
            print('-' * 10)
            early_stopping_ddi(val_epoch_loss, ddi_model)
            if early_stopping_ddi.early_stop:
                print("Early stopping")
                break

            test_preds = []
            test_labels = []
            ddi_testset = []              
            ddi_model.eval()
            with torch.no_grad():
                test_running_loss = 0.0
                test_num_preds = 0

                for j in range(0,ddi_test_num):
                    x = ddi_test_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    test_label = ddi_test_lab[j]
                    new_label = [i for x in test_label for i in x]
                    test_labels.extend(new_label)
                    test_label = torch.FloatTensor(test_label)
                    test_label = Variable(test_label).to(device)
                    test_outputs = ddi_model(x)
                    tmp_dis_testset = test_outputs.to(torch.device("cpu")).detach().numpy()
                    ddi_testset.append(tmp_dis_testset)
                    test_loss = criterion(test_outputs, test_label).to(device)
                    test_running_loss += test_loss.item()*x.size(0)
                    test_num_preds += x.size(0)
                    test_outputs = test_outputs.view(-1).tolist()
                    test_preds.extend(test_outputs)
                        
                test_epoch_loss = test_running_loss/test_num_preds
            test_auc_val = auc_score(test_labels, test_preds)
            print('DDI prediction test_auc: {:.4f}'.format(test_auc_val))
            print('-' * 10)
        ddi_test_aucs.extend([test_auc_val])
        ddi_testset = np.array(ddi_testset).reshape((-1,707))

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]
        train_drugs, train_Y = prepare_interaction_pairs_dse(dsevec, Y, trrows, trcols)
       
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        val_drugs, val_Y = prepare_interaction_pairs_dse(dsevec, Y, terows, tecols)

        testrows = label_row_inds[testinds]
        testcols = label_col_inds[testinds]
        test_drugs, test_Y = prepare_interaction_pairs_dse(dsevec, Y, testrows, testcols)

        tmp_drugs = []
        for i in range(0, len(test_drugs)):
            test_drugs[i][300:] = 0
            tmp_drugs.append(list(test_drugs[i][:300]))
        tmp_drugs = np.stack(tmp_drugs)

        dse_dat, dse_lab, dse_num = loadDataset_dse(train_drugs, train_Y, batchsz)
        dse_val_dat, dse_val_lab, dse_val_num = loadDataset_dse(val_drugs, val_Y, batchsz)
        dse_test_dat, dse_test_lab, dse_test_num = loadDataset_dse(test_drugs, test_Y, batchsz)

        dse_model = side_model(4492)
        dse_model.to(device)
        criterion = nn.BCELoss()
        learning_rate = 1e-5
        optimizer = optim.Adam(dse_model.parameters(), lr=learning_rate)

        auc_vals = 0.0
        dse_epoch = 70
        early_stopping_dse = EarlyStopping_dse(verbose=True, fold_info=foldind, epoch_info=dse_epoch, log_info=tmp_log)
        for i in range(dse_epoch):
            dse_model.train()
            print('Epoch {}/{}'.format(i, dse_epoch - 1))

            running_loss = 0.0
            num_preds_dti = 0
            running_loss_ddi = 0.0
            num_preds_ddi = 0

            running_loss = 0.0
            for j in range(0,dse_num):
                x = dse_dat[j]
                x = torch.from_numpy(x).float()
                x = Variable(x).to(device)

                label = dse_lab[j]
                label = torch.FloatTensor(label)
                label = Variable(label).to(device)

                outputs = dse_model(x)

                loss = criterion(outputs, label).to(device)
                for name, param in dse_model.named_parameters():
                    if 'weight' in name:
                        L1_1 = Variable(param, requires_grad=True)
                        L1_2 = torch.norm(L1_1, 2)
                        L1_3 = 10e-4*L1_2
                        loss = loss + L1_3
                running_loss += loss.item()*x.size(0)

                num_preds_dti += x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss0 = running_loss/num_preds_dti

            val_preds = []
            val_labels = []

            val_preds_ddi = []
            val_labels_ddi = []

            dse_model.eval()
            with torch.no_grad():
                for param in dse_model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                val_running_loss = 0.0
                val_num_preds = 0
                val_running_loss_ddi = 0.0
                val_num_preds_ddi = 0
                for j in range(0,dse_val_num):
                    x = dse_val_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    val_label = dse_val_lab[j]
                    new_label = [i for x in val_label for i in x]
                    val_labels.extend(new_label)

                    val_label = torch.FloatTensor(val_label)
                    val_label = Variable(val_label).to(device)

                    val_outputs = dse_model(x)

                    val_loss = criterion(val_outputs, val_label).to(device)
                    val_running_loss += val_loss.item()*x.size(0)
                    val_num_preds += x.size(0)
                    

                    val_outputs = val_outputs.view(-1).tolist()

                    val_preds.extend(val_outputs)

                val_epoch_loss = val_running_loss/val_num_preds
            auc_val = auc_score(val_labels, val_preds)
 
            print('DSE prediction training loss: {:.4f}, val_loss : {:.4f}, val_auc: {:.4f}'.format(epoch_loss0, val_epoch_loss, auc_val))
            print('-' * 10)
            early_stopping_dse(val_epoch_loss, dse_model)
            if early_stopping_dse.early_stop:
                print("Early stopping")
                break

            test_preds = []
            test_labels = []
            dse_testset = []              

            dse_model.eval()
            with torch.no_grad():
                test_running_loss = 0.0
                test_num_preds = 0

                for j in range(0,dse_test_num):
                    x = dse_test_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    test_label = dse_test_lab[j]
                    new_label = [i for x in test_label for i in x]
                    test_labels.extend(new_label)

                    test_label = torch.FloatTensor(test_label)
                    test_label = Variable(test_label).to(device)

                    test_outputs = dse_model(x)
                    tmp_dse_testset = test_outputs.to(torch.device("cpu")).detach().numpy()
                    dse_testset.append(tmp_dse_testset)

                    test_loss = criterion(test_outputs, test_label).to(device)
                    test_running_loss += test_loss.item()*x.size(0)
                    test_num_preds += x.size(0)

                    test_outputs = test_outputs.view(-1).tolist()
                    test_preds.extend(test_outputs)
                        
                test_epoch_loss = test_running_loss/test_num_preds
            test_auc_val = auc_score(test_labels, test_preds)
            print('DSE prediction test_auc: {:.4f}'.format(test_auc_val))
            print('-' * 10)
        dse_test_aucs.extend([test_auc_val])
        dse_testset = np.array(dse_testset).reshape((-1,4192))

        XD_train = disvec
        XT_train = XT

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = disvec[trrows]
        XT_train = XT[trcols]

        train_drugs, train_Y = prepare_interaction_pairs_dis(disvec, Y, trrows, trcols)
       
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]

        val_drugs, val_Y = prepare_interaction_pairs_dis(disvec, Y, terows, tecols)

        testrows = label_row_inds[testinds]
        testcols = label_col_inds[testinds]

        test_drugs, test_Y = prepare_interaction_pairs_dis(disvec, Y, testrows, testcols)

        tmp_drugs = []
        for i in range(0, len(test_drugs)):
            test_drugs[i][300:] = 0
            tmp_drugs.append(list(test_drugs[i][:300]))
        tmp_drugs = np.stack(tmp_drugs)


        dis_dat, dis_lab, dis_num = loadDataset_dis(train_drugs, train_Y, batchsz)
        dis_val_dat, dis_val_lab, dis_val_num = loadDataset_dis(val_drugs, val_Y, batchsz)
        dis_test_dat, dis_test_lab, dis_test_num = loadDataset_dis(test_drugs, test_Y, batchsz)

        dis_model = disease_model(1812)
        dis_model.to(device)
        criterion = nn.BCELoss()
        learning_rate = 1e-5
        optimizer = optim.Adam(dis_model.parameters(), lr=learning_rate)

        auc_vals = 0.0
        dis_epoch = 1000
        early_stopping_dis = EarlyStopping_dis(verbose=True, fold_info=foldind, epoch_info=dis_epoch, log_info=tmp_log)
        for i in range(dis_epoch):
            dis_model.train()
            print('Epoch {}/{}'.format(i, dis_epoch - 1))

            running_loss = 0.0
            num_preds_dti = 0
            running_loss_ddi = 0.0
            num_preds_ddi = 0

            running_loss = 0.0
            for j in range(0,dis_num):
                x = dis_dat[j]
                x = torch.from_numpy(x).float()
                x = Variable(x).to(device)

                label = dis_lab[j]
                label = torch.FloatTensor(label)
                label = Variable(label).to(device)

                outputs = dis_model(x)

                loss = criterion(outputs, label).to(device)
                for name, param in dis_model.named_parameters():
                    if 'weight' in name:
                        L1_1 = Variable(param, requires_grad=True)
                        L1_2 = torch.norm(L1_1, 2)
                        L1_3 = 10e-4*L1_2
                        loss = loss + L1_3
                running_loss += loss.item()*x.size(0)

                num_preds_dti += x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss0 = running_loss/num_preds_dti

            val_preds = []
            val_labels = []

            val_preds_ddi = []
            val_labels_ddi = []

            dis_model.eval()
            with torch.no_grad():
                for param in dis_model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                val_running_loss = 0.0
                val_num_preds = 0
                val_running_loss_ddi = 0.0
                val_num_preds_ddi = 0
                for j in range(0,dis_val_num):
                    x = dis_val_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    val_label = dis_val_lab[j]
                    new_label = [i for x in val_label for i in x]
                    val_labels.extend(new_label)

                    val_label = torch.FloatTensor(val_label)
                    val_label = Variable(val_label).to(device)

                    val_outputs = dis_model(x)

                    val_loss = criterion(val_outputs, val_label).to(device)
                    val_running_loss += val_loss.item()*x.size(0)
                    val_num_preds += x.size(0)

                    val_outputs = val_outputs.view(-1).tolist()

                    val_preds.extend(val_outputs)

                val_epoch_loss = val_running_loss/val_num_preds
            auc_val = auc_score(val_labels, val_preds)
 
            print('DIS prediction training loss: {:.4f}, val_loss : {:.4f}, val_auc: {:.4f}'.format(epoch_loss0, val_epoch_loss, auc_val))
            print('-' * 10)
            early_stopping_dis(val_epoch_loss, dis_model)
            if early_stopping_dis.early_stop:
                print("Early stopping")
                break

            test_preds = []
            test_labels = []
            dis_testset = []              

            dis_model.eval()
            with torch.no_grad():
                test_running_loss = 0.0
                test_num_preds = 0

                for j in range(0,dis_test_num):
                    x = dis_test_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    test_label = dis_test_lab[j]
                    new_label = [i for x in test_label for i in x]
                    test_labels.extend(new_label)

                    test_label = torch.FloatTensor(test_label)
                    test_label = Variable(test_label).to(device)

                    test_outputs = dis_model(x)
                    tmp_dis_testset = test_outputs.to(torch.device("cpu")).detach().numpy()
                    dis_testset.append(tmp_dis_testset)

                    test_loss = criterion(test_outputs, test_label).to(device)
                    test_running_loss += test_loss.item()*x.size(0)
                    test_num_preds += x.size(0)

                    test_outputs = test_outputs.view(-1).tolist()
                    test_preds.extend(test_outputs)
                        
                test_epoch_loss = test_running_loss/test_num_preds
            test_auc_val = auc_score(test_labels, test_preds)
            print('DIS prediction test_auc: {:.4f}'.format(test_auc_val))
            print('-' * 10)
        
        dis_test_aucs.extend([test_auc_val])
        dis_testset = np.array(dis_testset).reshape((-1,1512))

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]
        train_prots, train_Y = prepare_interaction_pairs_pdis(pdisvec, Y, trrows, trcols)
       
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        val_prots, val_Y = prepare_interaction_pairs_pdis(pdisvec,  Y, terows, tecols)

        testrows = label_row_inds[testinds]
        testcols = label_col_inds[testinds]
        test_prots, test_Y = prepare_interaction_pairs_pdis(pdisvec,  Y, testrows, testcols)
        for i in range(0, len(test_prots)):
            test_prots[i][100:] = 0

        pdis_dat, pdis_lab, pdis_num = loadDataset_pdis(train_prots, train_Y, batchsz)
        pdis_val_dat, pdis_val_lab, pdis_val_num = loadDataset_pdis(val_prots, val_Y, batchsz)
        pdis_test_dat, pdis_test_lab, pdis_test_num = loadDataset_pdis(test_prots, test_Y, batchsz)

        pdis_model = pdisease_model(5703)
        pdis_model.to(device)
        criterion = nn.BCELoss()
        learning_rate = 1e-5
        optimizer = optim.Adam(pdis_model.parameters(), lr=learning_rate)

        auc_vals = 0.0
        pdis_epoch = 30
        early_stopping_dis = EarlyStopping_pdis(verbose=True, fold_info=foldind, epoch_info=pdis_epoch, log_info=tmp_log)
        for i in range(pdis_epoch):
            pdis_model.train()
            print('Epoch {}/{}'.format(i, pdis_epoch - 1))

            running_loss = 0.0
            num_preds_dti = 0
            running_loss_ddi = 0.0
            num_preds_ddi = 0

            running_loss = 0.0
            for j in range(0,pdis_num):
                x = pdis_dat[j]
                x = torch.from_numpy(x).float()
                x = Variable(x).to(device)

                label = pdis_lab[j]
                label = torch.FloatTensor(label)
                label = Variable(label).to(device)

                outputs = pdis_model(x)

                loss = criterion(outputs, label).to(device)
                for name, param in pdis_model.named_parameters():
                    if 'weight' in name:
                        L1_1 = Variable(param, requires_grad=True)
                        L1_2 = torch.norm(L1_1, 2)
                        L1_3 = 10e-4*L1_2
                        loss = loss + L1_3
                running_loss += loss.item()*x.size(0)

                num_preds_dti += x.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss0 = running_loss/num_preds_dti

            val_preds = []
            val_labels = []

            val_preds_ddi = []
            val_labels_ddi = []

            pdis_model.eval()
            with torch.no_grad():
                for param in pdis_model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                val_running_loss = 0.0
                val_num_preds = 0
                val_running_loss_ddi = 0.0
                val_num_preds_ddi = 0
                for j in range(0,pdis_val_num):
                    x = pdis_val_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    val_label = pdis_val_lab[j]
                    new_label = [i for x in val_label for i in x]
                    val_labels.extend(new_label)

                    val_label = torch.FloatTensor(val_label)
                    val_label = Variable(val_label).to(device)

                    val_outputs = pdis_model(x)

                    val_loss = criterion(val_outputs, val_label).to(device)
                    val_running_loss += val_loss.item()*x.size(0)
                    val_num_preds += x.size(0)
                    

                    val_outputs = val_outputs.view(-1).tolist()

                    val_preds.extend(val_outputs)

                val_epoch_loss = val_running_loss/val_num_preds
            auc_val = auc_score(val_labels, val_preds)
 
            print('PDIS prediction training loss: {:.4f}, val_loss : {:.4f}, val_auc: {:.4f}'.format(epoch_loss0, val_epoch_loss, auc_val))
            print('-' * 10)
            early_stopping_dis(val_epoch_loss, pdis_model)
            if early_stopping_dis.early_stop:
                print("Early stopping")
                break

            test_preds = []
            test_labels = []
            pdis_testset = []              

            pdis_model.eval()
            with torch.no_grad():
                test_running_loss = 0.0
                test_num_preds = 0

                for j in range(0,pdis_test_num):
                    x = pdis_test_dat[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    test_label = pdis_test_lab[j]
                    new_label = [i for x in test_label for i in x]
                    test_labels.extend(new_label)

                    test_label = torch.FloatTensor(test_label)
                    test_label = Variable(test_label).to(device)

                    test_outputs = pdis_model(x)
                    tmp_dis_testset = test_outputs.to(torch.device("cpu")).detach().numpy()
                    pdis_testset.append(tmp_dis_testset)

                    test_loss = criterion(test_outputs, test_label).to(device)
                    test_running_loss += test_loss.item()*x.size(0)
                    test_num_preds += x.size(0)

                    test_outputs = test_outputs.view(-1).tolist()
                    test_preds.extend(test_outputs)
                        
                test_epoch_loss = test_running_loss/test_num_preds
            test_auc_val = auc_score(test_labels, test_preds)
            print('PDIS prediction test_auc: {:.4f}'.format(test_auc_val))
            print('-' * 10)

        pdis_test_aucs.extend([test_auc_val])
        pdis_testset = np.array(pdis_testset).reshape((-1,5603))

        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols, 0)
        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols, 0)
        test_drugs, test_prots, test_Y = prepare_interaction_pairs(XD, XT,  Y, testrows, testcols, 0)

        #############################drugs start!
        tmp_drugs = []
        for i in range(0, len(test_drugs)):
            test_drugs[i][300:] = 0
            tmp_drugs.append(list(test_drugs[i][:300]))

        init_ddi_vec = [0 for i in range(707)]

        ddi_vec=[]
        for i in range(0, len(test_drugs)):
            tmp_ddi_vec = tmp_drugs[i]+init_ddi_vec
            ddi_vec.append(tmp_ddi_vec)

        ddi_model.eval()
        with torch.no_grad():
            for param in ddi_model.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
            pretrained_ddi_vec = []
            for i in range(0, len(test_drugs)):
                y= ddi_vec[i]
                y = torch.from_numpy(np.asarray(y)).float()
                y = Variable(y).to(device)
                y_output = ddi_model(y)
                y_output = y_output.detach().cpu().tolist()
                pretrained_ddi_vec.append(y_output)

        init_se_vec = [0 for i in range(4192)]

        se_vec=[]
        for i in range(0, len(test_drugs)):
            tmp_se_vec = tmp_drugs[i]+init_se_vec
            se_vec.append(tmp_se_vec)

        dse_model.eval()
        with torch.no_grad():
            for param in dse_model.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
            pretrained_dse_vec = []
            for i in range(0, len(test_drugs)):
                y= se_vec[i]
                y = torch.from_numpy(np.asarray(y)).float()
                y = Variable(y).to(device)
                y_output = dse_model(y)
                y_output = y_output.detach().cpu().tolist()
                pretrained_dse_vec.append(y_output)

        init_dis_vec = [0 for i in range(1512)]
        dis_vec=[]
        for i in range(0, len(test_drugs)):
            tmp_dis_vec = tmp_drugs[i]+init_dis_vec
            dis_vec.append(tmp_dis_vec)
        
        dis_model.eval()
        with torch.no_grad():
            for param in dis_model.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
            pretrained_dis_vec = []
            for i in range(0, len(test_drugs)):
                y= dis_vec[i]
                y = torch.from_numpy(np.asarray(y)).float()
                y = Variable(y).to(device)
                y_output = dis_model(y)
                y_output = y_output.detach().cpu().tolist()
                pretrained_dis_vec.append(y_output)
        
        
        additional_vec = []
        for i in range(0, len(test_drugs)):
            tmp_add_vec = pretrained_ddi_vec[i] + pretrained_dse_vec[i] + pretrained_dis_vec[i]
            additional_vec.append(tmp_add_vec)
        
        test_drugs = np.asarray(tmp_drugs)
        test_drugall = np.asarray(additional_vec)
        

        ###############################################train_drus
        tmp_drugs = []
        for i in range(0, len(train_drugs)):
            tmp_drugs.append(list(train_drugs[i][:300]))

        
        additional_vec = []
        for i in range(0, len(train_drugs)):
            tmp_add_vec = train_drugs[i][300:]
            additional_vec.append(tmp_add_vec)

        train_drugs = np.asarray(tmp_drugs)
        train_drugall = np.asarray(additional_vec)

        ########################################val_drugs
        tmp_drugs = []
        for i in range(0, len(val_drugs)):
            #val_drugs[i][300:] = 0
            tmp_drugs.append(list(val_drugs[i][:300]))
        
        
        additional_vec = []
        for i in range(0, len(val_drugs)):
            tmp_add_vec = val_drugs[i][300:]
            additional_vec.append(tmp_add_vec)

        val_drugs = np.asarray(tmp_drugs)
        val_drugall = np.asarray(additional_vec)

        scaler = MinMaxScaler()

        t_psim_vecs = []
        for i in range(0, len(test_prots)):
            t_psim_vecs.append(list(test_prots[i][1589:3078]))
        
        psim_vecs = []
        for i in range(0, len(test_prots)):
            sim_x = np.array(t_psim_vecs[i]).reshape(-1,1)
            tmp_x = scaler.fit_transform(sim_x)
            tmp_simvec = list(tmp_x.flat)
            psim_vecs.append(tmp_simvec)
        t_ppi_vecs = []
        for i in range(0, len(test_prots)):
            t_ppi_vecs.append(list(test_prots[i][100:1589]))
            
        tmp_proteins = []
        for i in range(0, len(test_prots)):
            #test_prots[i][100:] = 0
            tmp_proteins.append(list(test_prots[i][:100]))

        init_pdis_vec = [0 for i in range(5603)]
        pdis_vec=[]
        for i in range(0, len(test_prots)):
            tmp_dis_vec = tmp_proteins[i]+init_pdis_vec
            pdis_vec.append(tmp_dis_vec)

        pdis_model.eval()
        with torch.no_grad():
            for param in pdis_model.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
            pretrained_dis_vec = []
            for i in range(0, len(test_prots)):
                y= pdis_vec[i]
                y = torch.from_numpy(np.asarray(y)).float()
                y = Variable(y).to(device)
                y_output = pdis_model(y)
                y_output = y_output.detach().cpu().tolist()
                pretrained_dis_vec.append(y_output)
    
        additional_vec = []
        for i in range(0, len(test_prots)):
            tmp_add_vec = t_ppi_vecs[i] + psim_vecs[i] + pretrained_dis_vec[i]
            additional_vec.append(tmp_add_vec)

        test_prots = np.asarray(tmp_proteins)
        test_protall = np.asarray(additional_vec)

        #######################################train_proteins
        tmp_proteins = []
        for i in range(0, len(train_prots)):
            #train_prots[i][100:] = 0
            tmp_proteins.append(list(train_prots[i][:100]))

        init_pdis_vec = [0 for i in range(5603)]
        pdis_vec=[]
        for i in range(0, len(train_prots)):
            tmp_dis_vec = tmp_proteins[i]+init_pdis_vec
            pdis_vec.append(tmp_dis_vec)

        pdis_model.eval()
        with torch.no_grad():
            for param in pdis_model.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
            pretrained_dis_vec = []
            for i in range(0, len(train_prots)):
                y= pdis_vec[i]
                y = torch.from_numpy(np.asarray(y)).float()
                y = Variable(y).to(device)
                y_output = pdis_model(y)
                y_output = y_output.detach().cpu().tolist()
                pretrained_dis_vec.append(y_output)
    
        t_psim_vecs = []
        for i in range(0, len(train_prots)):
            t_psim_vecs.append(list(train_prots[i][1589:3078]))
        
        psim_vecs = []
        for i in range(0, len(train_prots)):
            sim_x = np.array(t_psim_vecs[i]).reshape(-1,1)
            tmp_x = scaler.fit_transform(sim_x)
            tmp_simvec = list(tmp_x.flat)
            psim_vecs.append(tmp_simvec)
        
        for i in range(0, len(train_prots)):
            
            train_prots[i][1589:3078] = psim_vecs[i]
            train_prots[i][3078:] = pretrained_dis_vec[i]

        additional_vec = []
        for i in range(0, len(train_prots)):
            tmp_add_vec = train_prots[i][100:]
            additional_vec.append(tmp_add_vec)
        
        train_prots = np.asarray(tmp_proteins)
        train_protall = np.asarray(additional_vec)

        ##################################val_proteins
        tmp_proteins = []
        for i in range(0, len(val_prots)):
            #val_prots[i][100:] = 0
            tmp_proteins.append(list(val_prots[i][:100]))
        
        init_pdis_vec = [0 for i in range(5603)]
        pdis_vec=[]
        for i in range(0, len(val_prots)):
            tmp_dis_vec = tmp_proteins[i]+init_pdis_vec
            pdis_vec.append(tmp_dis_vec)

        pdis_model.eval()
        with torch.no_grad():
            for param in pdis_model.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
            pretrained_dis_vec = []
            for i in range(0, len(val_prots)):
                y= pdis_vec[i]
                y = torch.from_numpy(np.asarray(y)).float()
                y = Variable(y).to(device)
                y_output = pdis_model(y)
                y_output = y_output.detach().cpu().tolist()
                pretrained_dis_vec.append(y_output)
    
        t_psim_vecs = []
        for i in range(0, len(val_prots)):
            t_psim_vecs.append(list(val_prots[i][1589:3078]))
        
        psim_vecs = []
        for i in range(0, len(val_prots)):
            sim_x = np.array(t_psim_vecs[i]).reshape(-1,1)
            tmp_x = scaler.fit_transform(sim_x)
            tmp_simvec = list(tmp_x.flat)
            psim_vecs.append(tmp_simvec)

        for i in range(0, len(val_prots)):
            val_prots[i][1589:3078] = psim_vecs[i]
            val_prots[i][3078:] = pretrained_dis_vec[i]

        additional_vec = []
        for i in range(0, len(val_prots)):
            tmp_add_vec = val_prots[i][100:]
            additional_vec.append(tmp_add_vec)
        
        val_prots = np.asarray(tmp_proteins)
        val_protall = np.asarray(additional_vec)


        dat_all, lab, num, ct = loadDataset_dti(train_drugs, train_prots, train_drugall, train_protall, train_Y, batchsz)
        val_dat_all, val_lab, val_num, val_ct = loadDataset_dti(val_drugs, val_prots, val_drugall, val_protall, val_Y, batchsz)
        test_dat_all, test_lab, test_num, test_ct = loadDataset_dti(test_drugs, test_prots, test_drugall, test_protall, test_Y, batchsz)

        dvec_size = 300
        #dvec=300, ddi:707, dse:4192, ddis:1512 = 6711
        pvec_size = 100
        #pvec=100, ppi:1489, psim:1489, pdis:5603 = 8681
       

        print('-' * 10)
        start = time.time()

        model = dta_model(dvec_size, pvec_size)
        model.to(device)
        criterion = nn.BCELoss()
        learning_rate = 1e-5
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        auc_vals = 0.0
        early_stopping = EarlyStopping(verbose=True, fold_info=foldind, epoch_info=epoch, log_info=tmp_log)
        for i in range(epoch):
            model.train()
            print('Epoch {}/{}'.format(i, epoch - 1))

            running_loss_dti = 0.0
            num_preds_dti = 0
            running_loss_ddi = 0.0
            num_preds_ddi = 0

            running_loss = 0.0
            for j in range(0,num):
                x = dat_all[j]
                x = torch.from_numpy(x).float()
                x = Variable(x).to(device)

                label = lab[j]
                label = torch.FloatTensor(label)
                label = Variable(label).to(device)
                label = label.view(len(label),1)

                outputs = model(x)

                loss_dti = criterion(outputs, label).to(device)
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        L1_1 = Variable(param, requires_grad=True)
                        L1_2 = torch.norm(L1_1, 2)
                        L1_3 = learning_rate*L1_2
                        loss_dti = loss_dti + L1_3
                running_loss_dti += loss_dti.item()*x.size(0)

                num_preds_dti += x.size(0)

                optimizer.zero_grad()
                loss_dti.backward()
                optimizer.step()

            epoch_loss0 = running_loss_dti/num_preds_dti

            val_preds = []
            val_labels = []

            val_preds_ddi = []
            val_labels_ddi = []

            model.eval()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                val_running_loss = 0.0
                val_num_preds = 0
                val_running_loss_ddi = 0.0
                val_num_preds_ddi = 0
                for j in range(0,val_num):
                    x = val_dat_all[j]
                    x = torch.from_numpy(x).float()
                    x = Variable(x).to(device)

                    val_label = val_lab[j]
                    val_labels.extend(val_label)

                    val_label = torch.FloatTensor(val_label)
                    val_label = Variable(val_label).to(device)
                    val_label = val_label.view(len(val_label),1)

                    val_outputs = model(x)

                    val_loss = criterion(val_outputs, val_label).to(device)
                    val_running_loss += val_loss.item()*x.size(0)
                    val_num_preds += x.size(0)

                    val_outputs = val_outputs.view(len(val_outputs)).tolist()

                    val_preds.extend(val_outputs)

                val_epoch_loss = val_running_loss/val_num_preds
            auc_val = auc_score(val_labels, val_preds)
 
            print('DTI training loss: {:.4f}, val_loss : {:.4f}, val_auc: {:.4f}'.format(epoch_loss0, val_epoch_loss, auc_val))
            print('-' * 10)
            early_stopping(val_epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("total time :", time.time()-start)


        test_preds = []
        test_labels = []              

        model.eval()
        with torch.no_grad():
            test_running_loss = 0.0
            test_num_preds = 0

            for j in range(0,test_num):
                x = test_dat_all[j]
                x = torch.from_numpy(x).float()
                x = Variable(x).to(device)

                test_label = test_lab[j]
                test_labels.extend(test_label)

                test_label = torch.FloatTensor(test_label)
                test_label = Variable(test_label).to(device)
                test_label = test_label.view(len(test_label),1)

                test_outputs = model(x)

                test_loss = criterion(test_outputs, test_label).to(device)
                test_running_loss += test_loss.item()*x.size(0)
                test_num_preds += x.size(0)

                test_outputs = test_outputs.view(len(test_outputs)).tolist()
                test_preds.extend(test_outputs)
                        
            test_epoch_loss = test_running_loss/test_num_preds

        saved_data = pd.DataFrame()
        saved_data['test_rows']=testrows
        saved_data['test_preds']=test_preds
        saved_data['test_Y']=test_Y
        path =  "/home/auc_of_drugs/" + str(foldind) + "_test_predicted_all.csv"
        saved_data.to_csv(path, header=True, index=False)

        uniq_dlist = np.unique(testrows)
                
        for qnum in range(len(uniq_dlist)):
            tmp_dataset = saved_data.loc[saved_data['test_rows']==uniq_dlist[qnum]]
            tmp_labels = tmp_dataset['test_Y'].values
            tmp_preds = tmp_dataset['test_preds'].values
            tmp_num = np.count_nonzero(tmp_labels)
            if tmp_num != 0:
                tmp_auc = auc_score(tmp_labels, tmp_preds)
                tmp_length = len(tmp_dataset)
                tmp_posnum = len(tmp_dataset)/2
                tmp_list = [foldind, uniq_dlist[qnum], tmp_auc, tmp_posnum, tmp_length]
            else:
                tmp_auc = 'NaN'
                tmp_length = len(tmp_dataset)
                tmp_posnum = len(tmp_dataset)/2
                tmp_list = [foldind, uniq_dlist[qnum], tmp_auc, tmp_posnum, tmp_length]

        test_auc_val = auc_score(test_labels, test_preds)
        print('DTI test_auc: {:.4f}'.format(test_auc_val))
        print('-' * 10)

        plt.figure()
        logs = FLAGS.output_path
        logs = logs.split('/')
        logs = logs[1]
        logs = logs.replace('.', '_')
        path =  "/home/plots_all/DTI_1_1_fold" + str(foldind) + "_epoch_" +str(epoch)+ "_"+str(logs) + ".png"
        fig_title = "DTI(1:1) Fold"+ str(foldind)+ " epoch " +str(epoch)
        group = test_labels
        value = test_preds
        data = {'state':group, 'value':value}
        df = pd.DataFrame(data)
        df.groupby('state').value.hist(bins=200, alpha=0.6)
        plt.ylim(0,10)
        plt.title(fig_title, size=15)
        plt.xlabel('sigmoid value', size=15)
        plt.ylabel('count', size=15)
        plt.legend(['0','1'])
        plt.savefig(path)
        plt.close()

        logging("DTI: Fold = %d, AUC_val = %f, AUC_test = %f, BCE = %f, BCE_val = %f, BCE_test = %f" % (foldind, auc_val, test_auc_val, epoch_loss0, val_epoch_loss, test_epoch_loss), FLAGS)

        all_predictions[0][foldind] = test_auc_val
        all_losses[0][foldind]= test_epoch_loss


    ddi_avg_value = statistics.mean(ddi_test_aucs)
    logging("---DDI Prediction Results-----", FLAGS)
    logging("DDI test Performance AUC", FLAGS)
    logging(ddi_test_aucs, FLAGS)
    logging("DDI avg = %f" % (ddi_avg_value), FLAGS)

    dse_avg_value = statistics.mean(dse_test_aucs)
    logging("---Side-effect Prediction Results-----", FLAGS)
    logging("DDSE test Performance AUC", FLAGS)
    logging(dse_test_aucs, FLAGS)
    logging("DDSE avg = %f" % (dse_avg_value), FLAGS)

    dis_avg_value = statistics.mean(dis_test_aucs)
    logging("---Disease Prediction Results-----", FLAGS)
    logging("DDIS test Performance AUC", FLAGS)
    logging(dis_test_aucs, FLAGS)
    logging("DDIS avg = %f" % (dis_avg_value), FLAGS)

    pdis_avg_value = statistics.mean(pdis_test_aucs)
    logging("---Protein-disease Prediction Results-----", FLAGS)
    logging("PDIS test Performance AUC", FLAGS)
    logging(pdis_test_aucs, FLAGS)
    logging("PDIS avg = %f" % (pdis_avg_value), FLAGS)

        
    return  all_predictions, all_losses


def prepare_interaction_pairs(XD, XT,  Y, rows, cols, state_value):
    drugs = []
    targets = []
    targetscls = []
    affinity=[] 
    
    if state_value==0:
        for pair_ind in range(len(rows)):
            drug = XD[rows[pair_ind]]
            drugs.append(drug)

            target=XT[cols[pair_ind]]
            targets.append(target)

            affinity.append(Y[rows[pair_ind],cols[pair_ind]])

        drug_data = np.stack(drugs)
        target_data = np.stack(targets)
    elif state_value==1:
        for pair_ind in range(len(rows)):
            drug = XD[rows[pair_ind]]
            drugs.append(drug)
            target=XT[cols[pair_ind]]
            targets.append(target)
            affinity.append(Y[rows[pair_ind],cols[pair_ind]])
        drug_data = np.stack(drugs)
        target_data = np.stack(targets)
    else:
        print("Error of prepare_interaction_pairs")

    return drug_data,target_data,  affinity

def prepare_interaction_pairs_dis(XD, Y, rows, cols):
    drugs = []
    affinity=[] 
    
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)
        affinity.append(drug[300:].tolist())
    drug_data = np.stack(drugs)

    return drug_data, affinity

def prepare_interaction_pairs_dse(XD, Y, rows, cols):
    drugs = []
    affinity=[] 
    
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)
        affinity.append(drug[300:].tolist())

    drug_data = np.stack(drugs)


    return drug_data, affinity

def prepare_interaction_pairs_ddi(XD, Y, rows, cols):
    drugs = []
    affinity=[] 
    
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)
        affinity.append(drug[300:].tolist())

    drug_data = np.stack(drugs)

    return drug_data, affinity

def prepare_interaction_pairs_pdis(XT, Y, rows, cols):
    targets = []
    affinity=[] 
    
    for pair_ind in range(len(cols)):
        target=XT[cols[pair_ind]]
        targets.append(target)
        affinity.append(target[100:].tolist())

    target_data = np.stack(targets)

    return target_data,  affinity

def DTI_prediction(FLAGS):
    dataset = DataSet(fpath = FLAGS.dataset_path)

    dvec, pvec, Y_DTI, Y_DDI, Y_DSIE, Y_DDIS, Y_PPI, Y_PSIM, Y_PDIS = dataset.load_vectors(FLAGS)

    Y_DTI = np.asarray(Y_DTI)
    Y_DDI = np.asarray(Y_DDI)
    Y_DSIE = np.asarray(Y_DSIE)
    Y_DDIS = np.asarray(Y_DDIS)
    dvec = np.asarray(dvec)
    ddivec = np.concatenate((dvec, Y_DDI), axis=1)
    disvec = np.concatenate((dvec, Y_DDIS), axis=1)
    dsevec = np.concatenate((dvec, Y_DSIE), axis=1)
    dvec = np.concatenate((dvec, Y_DDI), axis=1)
    dvec = np.concatenate((dvec, Y_DSIE), axis=1)
    dvec = np.concatenate((dvec, Y_DDIS), axis=1)

    Y_PPI = np.asarray(Y_PPI)
    Y_PSIM = np.asarray(Y_PSIM)
    Y_PDIS = np.asarray(Y_PDIS)
    pvec = np.asarray(pvec)
    pdisvec = np.concatenate((pvec, Y_PDIS), axis=1)
    pvec = np.concatenate((pvec, Y_PPI), axis=1)
    pvec = np.concatenate((pvec, Y_PSIM), axis=1)
    pvec = np.concatenate((pvec, Y_PDIS), axis=1)

    alldat = dataset.load_all_dataset(FLAGS)
    all_dat_list = alldat.tolist()
    row_index = []
    col_index = []
    for i in range(len(alldat)):
        tmp_ridx = all_dat_list[i][0]
        tmp_cidx = all_dat_list[i][1]
        row_index.extend([tmp_ridx])
        col_index.extend([tmp_cidx])
    
    label_row_inds, label_col_inds = row_index, col_index
    label_row_inds = np.asarray(label_row_inds)
    label_col_inds = np.asarray(label_col_inds)


    train_index, valid_index, test_index = dataset.load_folds(FLAGS)
    
    foldinds = 10
    all_predictions, all_losses = cross_validation_DTI(dvec, pvec, Y_DTI, ddivec, dsevec, disvec, pdisvec, label_row_inds, label_col_inds, train_index, valid_index, test_index, FLAGS)
    

    testperfs = []
    testloss= []

    avgperf = 0.

    for test_foldind in range(0, foldinds):
        foldperf = all_predictions[0][test_foldind]
        foldloss = all_losses[0][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_index)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("DTI: test Performance AUC", FLAGS)
    logging(testperfs, FLAGS)
    logging("DTI: test Performance BCE", FLAGS)
    logging(testloss, FLAGS)


if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.output_path = "unseen_DPall_"+FLAGS.output_path + str(time.time()) + "/"

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    DTI_prediction(FLAGS)
