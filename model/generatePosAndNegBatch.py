# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:51:22 2020

@author: zjs
"""
import math
import torch
import pickle
import numpy as np
import random as rd
from torch.utils.data import Dataset

class Train_dataset(Dataset):

    def __init__(self, Train2id, Traindict, numOfTrain, numOfEntity, numOfRelation, ns):
        self.train2id = Train2id
        self.train_dict = Traindict
        self.numOfTrain = numOfTrain
        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.ns = ns

        rd.seed(2020)

        self.sample = 3
        self.global_sample = 1
        self.time_window_size = 5

        self.all_entity = set([ i for i in range(self.numOfEntity)])

    def __len__(self):
        return self.numOfTrain

    def __getitem__(self, item):
        sample =  self.generateTrainsample(item)
        return sample

    def find_times(self, e_dict, time):
        time_list = np.array(list(e_dict.keys()))
        times = list(time_list[time_list < time])
        times.extend([-1]*self.time_window_size)
        '''
        if len(times) == 0:
            times.extend([-1]*self.time_window_size)
        else:
            times.extend(np.random.choice(times, self.time_window_size))
        '''
        return times[0:self.time_window_size]

    def generateTrainsample(self, Id):
        tmpHead = self.train2id["s"][Id]# 1*1
        tmpRelation = self.train2id["r"][Id]
        tmpTail = self.train2id["o"][Id]
        tmpTime = self.train2id["t"][Id][3]

        tmpHead_d = len(self.train_dict["t2ee"][tmpTime][tmpHead])
        tmpTail_d = len(self.train_dict["t2ee"][tmpTime][tmpTail])
        tmpTime_m = 2*len(self.train_dict["t2ee"][tmpTime].keys())#self.train2id["t"].count(tmpTime)

        neg_s_set = self.all_entity# - set([tmpHead]) - set(self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime])
        #neg_s_set = (self.train_dict["ee2r"][tmpHead].keys()) | set(rd.sample(set(self.train_dict["t2ee"][tmpTime].keys()), 2*self.ns)) - set(self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime]) - set([tmpHead])
        ns = rd.sample(list(neg_s_set), self.ns)# 1*ns

        neg_o_set = self.all_entity# - set([tmpTail]) -  set(self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime])
        #neg_o_set = (self.train_dict["ee2r"][tmpTail].keys()) | set(rd.sample(set(self.train_dict["t2ee"][tmpTime].keys()), 2*self.ns)) - set(self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime]) - set([tmpTail]) 
        no = rd.sample(list(neg_o_set), self.ns)
        
        p_s_o_r = self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime]
        p_s_o_r.extend([self.numOfRelation]*self.sample)
        p_s_o_r = p_s_o_r[0:self.sample]

        s_h_r = [] #1*time_window_size*relation_num(sample)
        s_h_e = [] #1*time_window_size*relation_num(sample)*entity_num(sample)
        s_h_t = [] #1*time_window_size

        o_h_r = [] #1*time_window_size*relation_num(sample)
        o_h_e = [] #1*time_window_size*relation_num(sample)*entity_num(sample)
        o_h_t = [] #1*time_window_size


        s_h_dict = self.train_dict["etre"][tmpHead]
        s_h_times = self.find_times(s_h_dict, tmpTime)
        for time in s_h_times:
            if time < 0:
                s_h_e.append([[self.numOfEntity]*self.sample]*self.sample)
                s_h_r.append([self.numOfRelation]*self.sample)
                s_h_t.append(tmpTime)
            else:
                s_h_r_tmp = list(s_h_dict[time].keys())
                if len(s_h_r_tmp) < self.sample:
                    s_h_r_tmp.extend([self.numOfRelation]*self.sample)
                    s_h_r.append(s_h_r_tmp[0:self.sample])
                else:
                    s_h_r.append(rd.sample(s_h_r_tmp, self.sample))

                s_h_e_list = []
                for rel in s_h_r_tmp[0:self.sample]:
                    if rel != self.numOfRelation:
                        s_h_e_tmp = list(s_h_dict[time][rel])
                        s_h_e_tmp.extend([self.numOfEntity]*self.sample)
                        s_h_e_list.append(s_h_e_tmp[0:self.sample])
                    else:
                        s_h_e_list.append([self.numOfEntity]*self.sample)
                s_h_e.append(s_h_e_list)

                s_h_t.append(time)


        o_h_dict = self.train_dict["etre"][tmpTail]
        o_h_times = self.find_times(o_h_dict, tmpTime)
        for time in o_h_times:
            if time < 0:
                o_h_e.append([[self.numOfEntity]*self.sample]*self.sample)
                o_h_r.append([self.numOfRelation]*self.sample)
                o_h_t.append(tmpTime)
            else:
                o_h_r_tmp = list(o_h_dict[time].keys())
                if len(o_h_r_tmp) < self.sample:
                    o_h_r_tmp.extend([self.numOfRelation]*self.sample)
                    o_h_r.append(o_h_r_tmp[0:self.sample])
                else:
                    o_h_r.append(rd.sample(o_h_r_tmp, self.sample))

                o_h_e_list = []
                for rel in o_h_r_tmp[0:self.sample]:
                    if rel != self.numOfRelation:
                        o_h_e_tmp = list(o_h_dict[time][rel])
                        o_h_e_tmp.extend([self.numOfEntity]*self.sample)
                        o_h_e_list.append(o_h_e_tmp[0:self.sample])
                    else:
                        o_h_e_list.append([self.numOfEntity]*self.sample)
                o_h_e.append(o_h_e_list)

                o_h_t.append(time)

        return [tmpHead, tmpRelation, tmpTail, tmpTime, torch.LongTensor(ns), torch.LongTensor(no), \
                torch.LongTensor(s_h_r), torch.LongTensor(s_h_e), torch.LongTensor(s_h_t), \
                torch.LongTensor(o_h_r), torch.LongTensor(o_h_e), torch.LongTensor(o_h_t), \
                tmpHead_d, tmpTail_d, tmpTime_m, torch.LongTensor(p_s_o_r), \
                ]



class Test_dataset(Dataset):

    def __init__(self, Test2id, numOfTest, time_window_size, numOfEntity, numOfRelation, Traindict, dataset):
        self.Test2id = Test2id
        self.train_dict = Traindict
        self.numOfTest = numOfTest
        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation

        rd.seed(2020)

        self.sample = 3
        self.time_window_size = 5

    def __len__(self):
        return self.numOfTest

    def __getitem__(self, item):
        sample = self.generateTestsample(item)
        return sample

    def find_times(self, e_dict, time):
        time_list = np.array(list(e_dict.keys()))
        times = list(time_list[time_list < time])
        times.extend([-1]*self.time_window_size)

        return times[0:self.time_window_size]

    def generateTestsample(self, item):
        tmpHead = self.Test2id["s"][item]
        tmpRelation = self.Test2id["r"][item]
        tmpTail = self.Test2id["o"][item]
        tmpTime = self.Test2id["t"][item][3]

        s_h_e = []
        s_h_r = []
        s_h_t = []

        o_h_e = []
        o_h_r = []
        o_h_t = []

        all_h_e = []
        all_h_r = []
        all_h_t = []

        s_h_dict = self.train_dict["etre"][tmpHead]
        s_h_times = self.find_times(s_h_dict, tmpTime)
        for time in s_h_times:
            if time < 0:
                s_h_e.append([[self.numOfEntity]*self.sample]*self.sample)
                s_h_r.append([self.numOfRelation]*self.sample)
                s_h_t.append(tmpTime)
            else:
                s_h_r_tmp = list(s_h_dict[time].keys())
                if len(s_h_r_tmp) < self.sample:
                    s_h_r_tmp.extend([self.numOfRelation]*self.sample)
                    s_h_r.append(s_h_r_tmp[0:self.sample])
                else:
                    s_h_r.append(rd.sample(s_h_r_tmp, self.sample))

                s_h_e_list = []
                for rel in s_h_r_tmp[0:self.sample]:
                    if rel != self.numOfRelation:
                        s_h_e_tmp = list(s_h_dict[time][rel])
                        s_h_e_tmp.extend([self.numOfEntity]*self.sample)
                        s_h_e_list.append(s_h_e_tmp[0:self.sample])
                    else:
                        s_h_e_list.append([self.numOfEntity]*self.sample)
                s_h_e.append(s_h_e_list)

                s_h_t.append(time)


        o_h_dict = self.train_dict["etre"][tmpTail]
        o_h_times = self.find_times(o_h_dict, tmpTime)
        for time in o_h_times:
            if time < 0:
                o_h_e.append([[self.numOfEntity]*self.sample]*self.sample)
                o_h_r.append([self.numOfRelation]*self.sample)
                o_h_t.append(tmpTime)
            else:
                o_h_r_tmp = list(o_h_dict[time].keys())
                if len(o_h_r_tmp) < self.sample:
                    o_h_r_tmp.extend([self.numOfRelation]*self.sample)
                    o_h_r.append(o_h_r_tmp[0:self.sample])
                else:
                    o_h_r.append(rd.sample(o_h_r_tmp, self.sample))

                o_h_e_list = []
                for rel in o_h_r_tmp[0:self.sample]:
                    if rel != self.numOfRelation:
                        o_h_e_tmp = list(o_h_dict[time][rel])
                        o_h_e_tmp.extend([self.numOfEntity]*self.sample)
                        o_h_e_list.append(o_h_e_tmp[0:self.sample])
                    else:
                        o_h_e_list.append([self.numOfEntity]*self.sample)
                o_h_e.append(o_h_e_list)

                o_h_t.append(time)

        return [tmpHead, tmpRelation, tmpTail, tmpTime, \
                torch.LongTensor(s_h_r), torch.LongTensor(s_h_e), torch.LongTensor(s_h_t), \
                torch.LongTensor(o_h_r), torch.LongTensor(o_h_e), torch.LongTensor(o_h_t), \
                ]

        
        