# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:15:13 2020

@author: zjs
"""

import os
import torch
import pickle
import numpy as np
import random as rd
from datetime import date

class Data_process:
    
    def __init__(self, dataset):
        self.time_num = 0
        self.entity_num = 0
        self.relation_num = 0
        self.train_num = 0
        self.valid_num = 0
        self.test_num = 0
        self.train_dict = {}
        
        self.Train2id = {}
        self.Test2id = {}
        self.Valid2id = {}

        self.relation_sample = 3
        self.entity_sample = 5
        self.time_window_size = 5

        self.base_path = "../dataset/"+dataset+"/"
        self.titan_path = "/titan_data2/zhangjs/"+dataset+"/"+str(self.time_window_size)+"/"
        
        self.read_stat()
        self.read_train_dict()
        
        self.read_train()
        self.read_valid()
        self.read_test()
        
    def read_stat(self):
        path = self.base_path+"stat.txt"
        inputData = open(path)
        line = inputData.readline()
        
        self.entity_num = int(line.strip().split()[0])
        self.relation_num = int(line.strip().split()[1])
        self.time_num = int(line.strip().split()[2])
        self.train_num = int(line.strip().split()[3])
        self.valid_num = int(line.strip().split()[4])
        self.test_num = int(line.strip().split()[5])
        
        return self.entity_num, self.relation_num, self.time_num, self.train_num, self.valid_num, self.test_num

    def get_time(self, line):
        time = 0
        if "ICEWS14" in self.base_path:
            year = 1
            month = int(line.strip().split()[3].split('-')[1])
            day = int(line.strip().split()[3].split('-')[2])
            time = date(2014,month,day) - date(2014,1,1)
            ind = time.days
            time = [1, month, day, ind]

        if "ICEWS05" in self.base_path:
            year = int(line.strip().split()[3].split('-')[0])
            month = int(line.strip().split()[3].split('-')[1])
            day = int(line.strip().split()[3].split('-')[2])
            time = date(year,month,day) - date(2005,1,1)
            ind = time.days
            time = [year, month, day, ind]

        if "GDELT" in self.base_path:
            year = int(line.strip().split()[3].split('-')[0])
            month = int(line.strip().split()[3].split('-')[1])
            day = int(line.strip().split()[3].split('-')[2])
            time = date(year,month,day) - date(2015,4,1)
            ind = time.days
            time = [year, month, day, ind]

        return time
        
    def read_train_dict(self):
        if os.path.exists(self.base_path+"/train_dict_v4.pickle"):
            degree_input = open(self.base_path + "/train_dict_v4.pickle", "rb")
            self.train_dict.update(pickle.load(degree_input))
            degree_input.close()
            #print("train dict read finished")
            return self.train_dict
        
        path = self.base_path+"train.txt"
        inputData = open(path)
        lines = inputData.readlines()
        
        self.train_dict["er2e"] = {}
        self.train_dict["ee2r"] = {}
        self.train_dict["t2ee"] = {}
        self.train_dict["etre"] = {}
        for line in lines:
            tmpHead = int(line.strip().split()[0])
            tmpRelation = int(line.strip().split()[1])
            tmpTail = int(line.strip().split()[2])
            Times = self.get_time(line)
            tmpTime = Times[3]
            if tmpHead not in self.train_dict["etre"].keys():
                self.train_dict["etre"][tmpHead] = {}
                self.train_dict["etre"][tmpHead][tmpTime] = {}
                self.train_dict["etre"][tmpHead][tmpTime][tmpRelation] = set()
                self.train_dict["etre"][tmpHead][tmpTime][tmpRelation].add(tmpTail)
            else:
                if tmpTime not in self.train_dict["etre"][tmpHead].keys():
                    self.train_dict["etre"][tmpHead][tmpTime] = {}
                    self.train_dict["etre"][tmpHead][tmpTime][tmpRelation] = set()
                    self.train_dict["etre"][tmpHead][tmpTime][tmpRelation].add(tmpTail)
                else:
                    if tmpRelation not in self.train_dict["etre"][tmpHead][tmpTime].keys():
                        self.train_dict["etre"][tmpHead][tmpTime][tmpRelation] = set()
                        self.train_dict["etre"][tmpHead][tmpTime][tmpRelation].add(tmpTail)
                    else:
                        self.train_dict["etre"][tmpHead][tmpTime][tmpRelation].add(tmpTail)

            if tmpTail not in self.train_dict["etre"].keys():
                self.train_dict["etre"][tmpTail] = {}
                self.train_dict["etre"][tmpTail][tmpTime] = {}
                self.train_dict["etre"][tmpTail][tmpTime][tmpRelation] = set()
                self.train_dict["etre"][tmpTail][tmpTime][tmpRelation].add(tmpHead)
            else:
                if tmpTime not in self.train_dict["etre"][tmpTail].keys():
                    self.train_dict["etre"][tmpTail][tmpTime] = {}
                    self.train_dict["etre"][tmpTail][tmpTime][tmpRelation] = set()
                    self.train_dict["etre"][tmpTail][tmpTime][tmpRelation].add(tmpHead)
                else:
                    if tmpRelation not in self.train_dict["etre"][tmpTail][tmpTime].keys():
                        self.train_dict["etre"][tmpTail][tmpTime][tmpRelation] = set()
                        self.train_dict["etre"][tmpTail][tmpTime][tmpRelation].add(tmpHead)
                    else:
                        self.train_dict["etre"][tmpTail][tmpTime][tmpRelation].add(tmpHead)


            if tmpTime not in self.train_dict["t2ee"].keys():
                self.train_dict["t2ee"][tmpTime] = {}
                self.train_dict["t2ee"][tmpTime][tmpHead] = set()
                self.train_dict["t2ee"][tmpTime][tmpHead].add(tmpTail)
                self.train_dict["t2ee"][tmpTime][tmpTail] = set()
                self.train_dict["t2ee"][tmpTime][tmpTail].add(tmpHead)
            else:
                if tmpHead not in self.train_dict["t2ee"][tmpTime].keys():
                    self.train_dict["t2ee"][tmpTime][tmpHead] = set()
                    self.train_dict["t2ee"][tmpTime][tmpHead].add(tmpTail)
                else:
                    self.train_dict["t2ee"][tmpTime][tmpHead].add(tmpTail)

                if tmpTail not in self.train_dict["t2ee"][tmpTime].keys():
                    self.train_dict["t2ee"][tmpTime][tmpTail] = set()
                    self.train_dict["t2ee"][tmpTime][tmpTail].add(tmpHead)
                else:
                    self.train_dict["t2ee"][tmpTime][tmpTail].add(tmpHead)

            if tmpHead not in self.train_dict["er2e"].keys():
                self.train_dict["er2e"][tmpHead] = {}
                self.train_dict["er2e"][tmpHead][tmpRelation] = {}
                self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime] = []
                self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime].append(tmpTail)
            else:
                if tmpRelation not in self.train_dict["er2e"][tmpHead].keys():
                    self.train_dict["er2e"][tmpHead][tmpRelation] = {}
                    self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime] = []
                    self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime].append(tmpTail)
                else:
                    if tmpTime not in self.train_dict["er2e"][tmpHead][tmpRelation].keys():
                        self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime] = []
                        self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime].append(tmpTail)
                    else:
                        self.train_dict["er2e"][tmpHead][tmpRelation][tmpTime].append(tmpTail)

            if tmpTail not in self.train_dict["er2e"].keys():
                self.train_dict["er2e"][tmpTail] = {}
                self.train_dict["er2e"][tmpTail][tmpRelation] = {}
                self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime] = []
                self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime].append(tmpHead)
            else:
                if tmpRelation not in self.train_dict["er2e"][tmpTail].keys():
                    self.train_dict["er2e"][tmpTail][tmpRelation] = {}
                    self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime] = []
                    self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime].append(tmpHead)
                else:
                    if tmpTime not in self.train_dict["er2e"][tmpTail][tmpRelation].keys():
                        self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime] = []
                        self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime].append(tmpHead)
                    else:
                        self.train_dict["er2e"][tmpTail][tmpRelation][tmpTime].append(tmpHead)


            if tmpHead not in self.train_dict["ee2r"].keys():
                self.train_dict["ee2r"][tmpHead] = {}
                self.train_dict["ee2r"][tmpHead][tmpTail] = {}
                self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime] = []
                self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime].append(tmpRelation)
            else:
                if tmpTail not in self.train_dict["ee2r"][tmpHead].keys():
                    self.train_dict["ee2r"][tmpHead][tmpTail] = {}
                    self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime] = []
                    self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime].append(tmpRelation)
                else:
                    if tmpTime not in self.train_dict["ee2r"][tmpHead][tmpTail].keys():
                        self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime] = []
                        self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime].append(tmpRelation)
                    else:
                        self.train_dict["ee2r"][tmpHead][tmpTail][tmpTime].append(tmpRelation)

            if tmpTail not in self.train_dict["ee2r"].keys():
                self.train_dict["ee2r"][tmpTail] = {}
                self.train_dict["ee2r"][tmpTail][tmpHead] = {}
                self.train_dict["ee2r"][tmpTail][tmpHead][tmpTime] = []
                self.train_dict["ee2r"][tmpTail][tmpHead][tmpTime].append(tmpRelation)
            else:
                if tmpHead not in self.train_dict["ee2r"][tmpTail].keys():
                    self.train_dict["ee2r"][tmpTail][tmpHead] = {}
                    self.train_dict["ee2r"][tmpTail][tmpHead][tmpTime] = []
                    self.train_dict["ee2r"][tmpTail][tmpHead][tmpTime].append(tmpRelation)
                else:
                    if tmpTime not in self.train_dict["ee2r"][tmpTail][tmpHead].keys():
                        self.train_dict["ee2r"][tmpTail][tmpHead][tmpTime] = []
                        self.train_dict["ee2r"][tmpTail][tmpHead][tmpTime].append(tmpRelation)
                    else:
                        self.train_dict["ee2r"][tmpTail][tmpHead][tmpTime].append(tmpRelation)


        train_dict_path = self.base_path + "/train_dict_v4.pickle"
        Output = open(train_dict_path, "wb")
        pickle.dump(self.train_dict, Output)
        Output.close()
        #print("train dict read finished")
        
        return self.train_dict


    def read_train(self):
        if os.path.exists(self.base_path+"/train_v4.pickle"):
            data2id_input = open(self.base_path + "/train_v4.pickle", "rb")
            self.Train2id.update(pickle.load(data2id_input))
            #print(len(self.data2id["train"]))
            data2id_input.close()
            return self.Train2id
        
        path = self.base_path+"train.txt"
        data = open(path,encoding="UTF-8")
        lines = data.readlines()
        self.Train2id["s"] = []
        self.Train2id["r"] = []
        self.Train2id["o"] = []
        self.Train2id["t"] = []

        for line in lines:
            sbj = int(line.strip().split()[0])
            rel = int(line.strip().split()[1])
            obj = int(line.strip().split()[2])
            time = self.get_time(line)
            
            self.Train2id["s"].append(sbj)
            self.Train2id["r"].append(rel)
            self.Train2id["o"].append(obj)
            self.Train2id["t"].append(time)
            
        data2id_path = self.base_path + "/train_v4.pickle"
        Output = open(data2id_path, "wb")
        pickle.dump(self.Train2id, Output)
        Output.close()
        
        return self.Train2id
    
    def read_valid(self):
        if os.path.exists(self.base_path+"/valid_v4.pickle"):
            data2id_input = open(self.base_path + "/valid_v4.pickle", "rb")
            self.Valid2id.update(pickle.load(data2id_input))
            #print(len(self.data2id["train"]))
            data2id_input.close()
            return self.Valid2id
        
        path = self.base_path+"valid.txt"
        data = open(path,encoding="UTF-8")
        lines = data.readlines()
        self.Valid2id["s"] = []
        self.Valid2id["r"] = []
        self.Valid2id["o"] = []
        self.Valid2id["t"] = [] 
        for line in lines:
            sbj = int(line.strip().split()[0])
            rel = int(line.strip().split()[1])
            obj = int(line.strip().split()[2])
            time = self.get_time(line)
            
            self.Valid2id["s"].append(sbj)
            self.Valid2id["r"].append(rel)
            self.Valid2id["o"].append(obj)
            self.Valid2id["t"].append(time) 
            
        data2id_path = self.base_path + "/valid_v4.pickle"
        Output = open(data2id_path, "wb")
        pickle.dump(self.Valid2id, Output)
        Output.close()
        
        return self.Valid2id
    
    def read_test(self):
        
        if os.path.exists(self.base_path+"/test_v4.pickle"):
            data2id_input = open(self.base_path + "/test_v4.pickle", "rb")
            self.Test2id.update(pickle.load(data2id_input))
            #print(len(self.data2id["train"]))
            data2id_input.close()
            return 0
        
        path = self.base_path+"test.txt"
        data = open(path,encoding="UTF-8")
        lines = data.readlines()
        self.Test2id["s"] = []
        self.Test2id["r"] = []
        self.Test2id["o"] = []
        self.Test2id["t"] = []
        for line in lines:
            s = int(line.strip().split()[0])
            r = int(line.strip().split()[1])
            o = int(line.strip().split()[2])
            time = self.get_time(line)
            
            self.Test2id["s"].append(s)
            self.Test2id["r"].append(r)
            self.Test2id["o"].append(o)
            self.Test2id["t"].append(time)   
        
        data2id_path = self.base_path + "/test_v4.pickle"
        Output = open(data2id_path, "wb")
        pickle.dump(self.Test2id, Output)
        Output.close()
        
        return 0