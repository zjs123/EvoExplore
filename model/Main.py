# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:10:42 2020

@author: zjs
"""
import sys
import torch
import pickle
import argparse
import progressbar
import numpy as np
import random as rd
from Model import model
import torch.optim as optim
from Data_process import Data_process
from torch.utils.data import DataLoader
from generatePosAndNegBatch import Train_dataset, Test_dataset
from multiprocessing import Manager, Pool

class Main:
    
    def __init__(self, args):
        self.lr = args.lr
        self.ns = args.ns
        self.norm = args.norm
        self.layer = args.layer
        self.margin = args.margin
        self.hidden = args.hidden
        self.dataset = args.dataset
        self.numOfEpoch = args.numOfEpoch
        self.numOfBatches = 200
        self.relation_sample = 3
        self.entity_sample = 5
        self.time_window_size = 5

        self.result_path = "../dataset/"+self.dataset+"/"+"result.txt"
        
        Data = Data_process(self.dataset)
        
        self.numOfTrain = Data.train_num
        self.numOfValid = Data.valid_num
        self.numOfTest = Data.test_num 
        self.numOfTime = Data.time_num
        self.numOfEntity = Data.entity_num
        self.numOfRelation = Data.relation_num
        
        self.Train2id = Data.Train2id
        self.Test2id = Data.Test2id
        self.Valid2id = Data.Valid2id

        self.Traindict = Data.train_dict
      
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        #self.device = torch.device("cpu")
        
        self.model = model(self.numOfEntity, self.numOfRelation, self.hidden, self.ns)
        self.model.to(self.device)

        self.Train_Dataset = Train_dataset(self.Train2id, self.Traindict, self.numOfTrain, self.numOfEntity, self.numOfRelation, self.ns)
        self.Test_Dataset = Test_dataset(self.Test2id, self.numOfTest, self.time_window_size, self.numOfEntity, self.numOfRelation, self.Traindict, self.dataset)
        
        self.Train()
        #self.test()
        
    def Train(self):
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        dataLoader = DataLoader(self.Train_Dataset, int(self.numOfTrain/self.numOfBatches), shuffle = True, pin_memory = True, num_workers = 8)
        for epoch in range(self.numOfEpoch+1):
            #self.adjust_learning_rate(optimizer, epoch)
            epochLoss = 0
            batch_num = 0
            p = progressbar.ProgressBar(widgets = ["Epoch", str(epoch),":[", progressbar.Percentage(),"]", progressbar.Timer()], maxval = self.numOfBatches)
            p.start()
            for batch in dataLoader:
                p.update(batch_num)
                batch_num += 1
                optimizer.zero_grad()

                p_s = torch.LongTensor(batch[0]).to(self.device)
                p_r = torch.LongTensor(batch[1]).to(self.device)
                p_o = torch.LongTensor(batch[2]).to(self.device)
                p_t = torch.LongTensor(batch[3]).float().to(self.device)

                n_s_l = torch.LongTensor(batch[4]).to(self.device)
                n_o_l = torch.LongTensor(batch[5]).to(self.device)

                s_h_r = torch.LongTensor(batch[6]).to(self.device)
                s_h_e = torch.LongTensor(batch[7]).to(self.device)
                s_h_t = torch.LongTensor(batch[8]).float().to(self.device)

                o_h_r = torch.LongTensor(batch[9]).to(self.device)
                o_h_e = torch.LongTensor(batch[10]).to(self.device)
                o_h_t = torch.LongTensor(batch[11]).float().to(self.device)

                p_s_d = torch.LongTensor(batch[12]).float().to(self.device)
                p_o_d = torch.LongTensor(batch[13]).float().to(self.device)
                p_t_m = torch.LongTensor(batch[14]).float().to(self.device)

                p_s_o_r = torch.LongTensor(batch[15]).to(self.device)

                n_s_g = torch.LongTensor(batch[16]).to(self.device)
                n_o_g = torch.LongTensor(batch[17]).to(self.device)

                #print(N_t_e[0][0])
                batchLoss = self.model(p_s, p_r, p_o, p_t, n_s_l, n_o_l, \
						                s_h_r, s_h_e, s_h_t, \
						                o_h_r, o_h_e, o_h_t, \
						                p_s_d, p_o_d, p_t_m, \
                                        p_s_o_r, \
                                        n_s_g, n_o_g \
                                        )
                batchLoss.backward()
                optimizer.step()
                
                #print("Batch: " + str(batch_num) + " | Loss: " + str(batchLoss))
                epochLoss += batchLoss
            p.finish()   
            print("loss: " + str(float(epochLoss)))
            
            if epoch % 50 == 0 and epoch!=0 :
	            with torch.no_grad():
	                self.model.eval()
	                self.Test()
	                #self.write(self.model) 
	

    def write(self, model):
        #print "-----Writing Training Results to " + self.outAdd + "-----"
        transmit_path = "../dataset/" + self.dataset + "/model_no_sum.pickle"
        modelOutput = open(transmit_path, "wb")
        pickle.dump(model, modelOutput)
        modelOutput.close()

    def preread(self):
        modelInput = open("./dataset/"+self.dataset + "/model_score.pickle", "rb")
        self.model = pickle.load(modelInput)
        modelInput.close()

    def adjust_learning_rate(self, optimizer, epoch):
        lr = args.lr * (0.5 ** (epoch // 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr        
    
    def Test(self):
        MRR = 0
        H1 = 0
        H3 = 0
        H5 = 0
        H10 = 0
        dataLoader = DataLoader(self.Test_Dataset, 1, shuffle = False, pin_memory = True, num_workers = 6)
        num = 0
        p = progressbar.ProgressBar(widgets = ["Valid:", progressbar.Bar('*'), progressbar.Percentage(), "|", progressbar.Timer()], maxval = self.numOfTest//1 + 1)
        p.start()
        for test_sample in dataLoader:
            p.update(num)
            num += 1

            s = torch.LongTensor(test_sample[0]).to(self.device)
            r = torch.LongTensor(test_sample[1]).to(self.device)
            o = torch.LongTensor(test_sample[2]).to(self.device)
            t = torch.LongTensor(test_sample[3]).float().to(self.device)

            s_h_r = torch.LongTensor(test_sample[4]).to(self.device)
            s_h_e = torch.LongTensor(test_sample[5]).to(self.device)
            s_h_t = torch.LongTensor(test_sample[6]).float().to(self.device)

            o_h_r = torch.LongTensor(test_sample[7]).to(self.device)
            o_h_e = torch.LongTensor(test_sample[8]).to(self.device)
            o_h_t = torch.LongTensor(test_sample[9]).float().to(self.device)

            sub_MRR, sub_H1, sub_H3, sub_H5, sub_H10 = self.model.validate(s, r, o, t, \
															                s_h_r, s_h_e, s_h_t, \
															                o_h_r, o_h_e, o_h_t, \
															                self.Traindict)
            MRR += sub_MRR
            H1 += sub_H1
            H3 += sub_H3
            H5 += sub_H5
            H10 += sub_H10
        	
        p.finish()
        MRR = MRR/(2*self.numOfTest)
        H1 = H1/(2*self.numOfTest)
        H3 = H3/(2*self.numOfTest)
        H5 = H5/(2*self.numOfTest)
        H10 = H10/(2*self.numOfTest)

        print("valid MRR: "+str(MRR))
        print("valid H1: "+str(H1))
        print("valid H3: "+str(H3))
        print("valid H5: "+str(H5))
        print("valid H10: "+str(H10))

        return 0

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model")
    parser.add_argument("--hidden",dest="hidden",type=int,default=100)
    parser.add_argument("--lr", dest="lr", type=float, default=0.001)
    parser.add_argument("--ns", dest="ns", type=int, default=10)
    parser.add_argument("--margin",dest="margin",type=int,default=1)
    parser.add_argument("--norm", dest="norm", type=int, default=2)
    parser.add_argument("--layer", dest="layer", type=int, default=1)
    parser.add_argument("--dataset",dest="dataset",type=str,default="GDELT")
    parser.add_argument("--numOfEpoch",dest="numOfEpoch",type=int,default=50)
    
    args=parser.parse_args()
    Main(args)
        
