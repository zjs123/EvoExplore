# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:34:25 2020

@author: zjs
"""
import Loss
import torch
import numpy as np
import random as rd
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    
    def __init__(self, numOfEntity, numOfRelation, numOfhidden, ns):
        super(model,self).__init__()
        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.numOfhidden = numOfhidden
        self.norm = 2
        self.ns = ns
        self.prob = 0.0
        self.numOfCommunity = 6
        
        sqrtR = numOfhidden**0.5
        sqrtE = numOfhidden**0.5

        self.relation_embeddings = nn.Embedding(self.numOfRelation+1, self.numOfhidden, padding_idx=self.numOfRelation)
        #nn.init.xavier_uniform_(self.relation_embeddings.weight[0:self.numOfRelation])
        self.relation_embeddings.weight.data[0:self.numOfRelation] = torch.FloatTensor(self.numOfRelation, self.numOfhidden).uniform_(-1./sqrtR, 1./sqrtR)
        self.relation_embeddings.weight.data[0:self.numOfRelation] = F.normalize(self.relation_embeddings.weight.data[0:self.numOfRelation], 2, 1)

        self.entity_embeddings_y1 = nn.Embedding(self.numOfEntity+1, 30, padding_idx=self.numOfEntity)
        nn.init.xavier_uniform_(self.entity_embeddings_y1.weight[0:self.numOfEntity])

        self.entity_embeddings_m1 = nn.Embedding(self.numOfEntity+1, 30, padding_idx=self.numOfEntity)
        nn.init.xavier_uniform_(self.entity_embeddings_m1.weight[0:self.numOfEntity])
        
        self.entity_embeddings_d1 = nn.Embedding(self.numOfEntity+1, 40, padding_idx=self.numOfEntity)
        nn.init.xavier_uniform_(self.entity_embeddings_d1.weight[0:self.numOfEntity])

        self.entity_embeddings_y2 = nn.Embedding(self.numOfEntity+1, 30, padding_idx=self.numOfEntity)
        nn.init.xavier_uniform_(self.entity_embeddings_y2.weight[0:self.numOfEntity])

        self.entity_embeddings_m2 = nn.Embedding(self.numOfEntity+1, 30, padding_idx=self.numOfEntity)
        nn.init.xavier_uniform_(self.entity_embeddings_m2.weight[0:self.numOfEntity])
        
        self.entity_embeddings_d2 = nn.Embedding(self.numOfEntity+1, 40, padding_idx=self.numOfEntity)
        nn.init.xavier_uniform_(self.entity_embeddings_d2.weight[0:self.numOfEntity])

        self.entity_embeddings_v3 = nn.Embedding(self.numOfEntity+1, self.numOfhidden, padding_idx=self.numOfEntity)
        #nn.init.xavier_uniform_(self.entity_embeddings_v3.weight[0:self.numOfEntity])
        self.entity_embeddings_v3.weight.data[0:self.numOfEntity] = torch.FloatTensor(self.numOfEntity, self.numOfhidden).uniform_(-1./sqrtE, 1./sqrtE)
        self.entity_embeddings_v3.weight.data[0:self.numOfEntity] = F.normalize(self.entity_embeddings_v3.weight.data[0:self.numOfEntity], 2, 1)

        self.Transfer_R = nn.Linear(self.numOfhidden, self.numOfhidden, bias = False)
        self.Transfer_E = nn.Linear(self.numOfhidden, self.numOfhidden, bias = False)
        self.decay_rate = torch.nn.Parameter(data = torch.cuda.FloatTensor([[0.001]]), requires_grad = True)
        self.trade_off = 0#torch.nn.Parameter(data = torch.cuda.FloatTensor([[0.5]]), requires_grad = True)
        self.Transfer_SH = nn.Linear(self.numOfhidden, self.numOfhidden, bias = False)
        
        self.score_vector = torch.nn.Parameter(data = torch.cuda.FloatTensor(1, self.numOfhidden), requires_grad = True)
        nn.init.xavier_uniform_(self.score_vector)

        self.auxiliary_matrix = nn.Linear(self.numOfhidden, self.numOfCommunity, bias = False)
        self.ConvE =ConvKB(self.numOfhidden, 3, 1, 5, 0.0, 0.2)
        
    def forward(self, p_s, p_r, p_o, p_t, n_s_l, n_o_l, \
                s_h_r, s_h_e, s_h_t, \
                o_h_r, o_h_e, o_h_t, \
                p_s_d, p_o_d, p_t_m, \
                p_s_o_r, \
                n_s_g, n_o_g \
                ):

        local_loss = self.get_local_loss(p_s, p_r, p_o, p_t, n_s_l, n_o_l, \
                                            s_h_r, s_h_e, s_h_t, \
                                            o_h_r, o_h_e, o_h_t, \
                                            )

        s_h_e_flat = s_h_e.flatten(1) #batch_size*(time_window*sample*sample)
        o_h_e_flat = o_h_e.flatten(1) #batch_size*(time_window*sample*sample)

        global_loss = self.get_global_loss(p_s, p_o, p_t, \
                                           n_s_g, n_o_g, \
                                           s_h_e_flat, o_h_e_flat, \
                                           p_s_d, p_o_d, p_t_m, \
                                           p_s_o_r)

        #co_loss = self.get_co_loss()

        batchLoss = local_loss #+ 0.1*global_loss# + co_loss

        return batchLoss

    def get_time_embedding_A(self, ind, t, dim):
        embedding_y1 = self.entity_embeddings_y1(ind)
        embedding_m1 = self.entity_embeddings_m1(ind)
        embedding_d1 = self.entity_embeddings_d1(ind)

        embedding_y2 = self.entity_embeddings_y2(ind)
        embedding_m2 = self.entity_embeddings_m2(ind)
        embedding_d2 = self.entity_embeddings_d2(ind)
        
        #embedding_period = (torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1))
        #embedding_nonper = (torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1))
        embedding_period = F.normalize(torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1), 2, -1)
        embedding_nonper = F.normalize(torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1), 2, -1)
        #embedding_period = F.normalize(torch.sin(embedding_y1*(t + 1)), 2, -1)
        #embedding_nonper = F.normalize(torch.tanh(embedding_y2*(t + 1)), 2, -1)
        embedding_static = self.entity_embeddings_v3(ind)
        #embedding = F.normalize(embedding_period + embedding_nonper + embedding_static, 2 ,dim)
        #embedding = (embedding_period + embedding_nonper + embedding_static)
        embedding = embedding_static
        #embedding = torch.cat([embedding_period, embedding_nonper, embedding_static], dim)

        return embedding

    def get_time_embedding_B(self, ind, t, dim):
        embedding_y1 = self.entity_embeddings_y1(ind).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        embedding_m1 = self.entity_embeddings_m1(ind).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        embedding_d1 = self.entity_embeddings_d1(ind).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        embedding_y2 = self.entity_embeddings_y2(ind).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        embedding_m2 = self.entity_embeddings_m2(ind).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        embedding_d2 = self.entity_embeddings_d2(ind).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        #embedding_period = (torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1))
        #embedding_nonper = (torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1))
        embedding_period = F.normalize(torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1), 2, -1)
        embedding_nonper = F.normalize(torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1), 2, -1)
        #embedding_period = F.normalize(torch.sin(embedding_y1*(t + 1)), 2, -1)
        #embedding_nonper = F.normalize(torch.tanh(embedding_y2*(t + 1)), 2, -1)
        embedding_static = self.entity_embeddings_v3(ind).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        embedding_static = embedding_static.repeat(1, embedding_period.size()[1], embedding_period.size()[2], embedding_period.size()[3], 1)
        #embedding = F.normalize(embedding_period + embedding_nonper + embedding_static, 2 ,dim)
        #embedding = (embedding_period + embedding_nonper + embedding_static)
        embedding = embedding_static
        #embedding = torch.cat([embedding_period, embedding_nonper, embedding_static], dim)

        return embedding

    def get_time_embedding_C(self, ind, t, dim):
        embedding_y1 = self.entity_embeddings_y1(ind).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_m1 = self.entity_embeddings_m1(ind).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_d1 = self.entity_embeddings_d1(ind).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        embedding_y2 = self.entity_embeddings_y2(ind).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_m2 = self.entity_embeddings_m2(ind).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_d2 = self.entity_embeddings_d2(ind).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        #embedding_period = (torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1))
        #embedding_nonper = (torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1))
        embedding_period = F.normalize(torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1), 2, -1)
        embedding_nonper = F.normalize(torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1), 2, -1)
        #embedding_period = F.normalize(torch.sin(embedding_y1*(t + 1)), 2, -1)
        #embedding_nonper = F.normalize(torch.tanh(embedding_y2*(t + 1)), 2, -1)
        embedding_static = self.entity_embeddings_v3(ind).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_static = embedding_static.repeat(1, 1, embedding_period.size()[2], embedding_period.size()[3], embedding_period.size()[4], 1)
        #embedding = F.normalize(embedding_period + embedding_nonper + embedding_static, 2 ,dim)
        #embedding = (embedding_period + embedding_nonper + embedding_static)
        embedding = embedding_static
        #embedding = torch.cat([embedding_period, embedding_nonper, embedding_static], dim)

        return embedding       

    def get_time_embedding_D(self, t, dim):
        embedding_y1 = self.entity_embeddings_y1.weight.data[0:self.numOfEntity].unsqueeze(0)
        embedding_m1 = self.entity_embeddings_m1.weight.data[0:self.numOfEntity].unsqueeze(0)
        embedding_d1 = self.entity_embeddings_d1.weight.data[0:self.numOfEntity].unsqueeze(0)

        embedding_y2 = self.entity_embeddings_y2.weight.data[0:self.numOfEntity].unsqueeze(0)
        embedding_m2 = self.entity_embeddings_m2.weight.data[0:self.numOfEntity].unsqueeze(0)
        embedding_d2 = self.entity_embeddings_d2.weight.data[0:self.numOfEntity].unsqueeze(0)
        
        #embedding_period = (torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1))
        #embedding_nonper = (torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1))
        embedding_period = F.normalize(torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1), 2, -1)
        embedding_nonper = F.normalize(torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1), 2, -1)
        #embedding_period = F.normalize(torch.sin(embedding_y1*(t + 1)), 2, -1)
        #embedding_nonper = F.normalize(torch.tanh(embedding_y2*(t + 1)), 2, -1)
        embedding_static = self.entity_embeddings_v3.weight.data[0:self.numOfEntity].unsqueeze(0)
        embedding_static = embedding_static.repeat(embedding_period.size()[0], 1, 1)
        #embedding = F.normalize(embedding_period + embedding_nonper + embedding_static, 2 ,dim)
        #embedding = (embedding_period + embedding_nonper + embedding_static)
        embedding = embedding_static
        #embedding = torch.cat([embedding_period, embedding_nonper, embedding_static], dim)

        return embedding

    def get_time_embedding_E(self, t, dim):
        embedding_y1 = self.entity_embeddings_y1.weight.data[0:self.numOfEntity].unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_m1 = self.entity_embeddings_m1.weight.data[0:self.numOfEntity].unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_d1 = self.entity_embeddings_d1.weight.data[0:self.numOfEntity].unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        embedding_y2 = self.entity_embeddings_y2.weight.data[0:self.numOfEntity].unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_m2 = self.entity_embeddings_m2.weight.data[0:self.numOfEntity].unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_d2 = self.entity_embeddings_d2.weight.data[0:self.numOfEntity].unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        
        
        #embedding_period = (torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1))
        #embedding_nonper = (torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1))
        embedding_period = F.normalize(torch.cat([torch.sin(embedding_y1*(t//366 + 1)), torch.sin(embedding_m1*(t//30 + 1)), torch.sin(embedding_d1*(t%30 + 1))], -1), 2, -1)
        embedding_nonper = F.normalize(torch.cat([torch.tanh(embedding_y2*(t//366 + 1)), torch.tanh(embedding_m2*(t//30 + 1)), torch.tanh(embedding_d2*(t%30 + 1))], -1), 2, -1)
        #embedding_period = F.normalize(torch.sin(embedding_y1*(t + 1)), 2, -1)
        #embedding_nonper = F.normalize(torch.tanh(embedding_y2*(t + 1)), 2, -1)
        embedding_static = self.entity_embeddings_v3.weight.data[0:self.numOfEntity].unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        embedding_static = embedding_static.repeat(embedding_period.size()[0], 1, embedding_period.size()[2], embedding_period.size()[3], embedding_period.size()[4], 1)
        #embedding = F.normalize(embedding_period + embedding_nonper + embedding_static, 2 ,dim)
        #embedding = (embedding_period + embedding_nonper + embedding_static)
        embedding = embedding_static
        #embedding = torch.cat([embedding_period, embedding_nonper, embedding_static], dim)
        
        return embedding

    def masked_softmax(self, A, B, k, dim):
        
        A = A.float()
        A_max = torch.max(A,dim=dim,keepdim=True)[0]
        A_exp = torch.exp(A-A_max)
        A_exp = A_exp * (B != k).float()
        Sum = torch.sum(A_exp,dim=dim,keepdim=True)
        Sum = Sum+1
        score = A_exp / Sum
        
        '''
        A_masked = A*(B != k).float() + (B == k).float()*(-1e6)
        A_exp = torch.exp(A_masked)
        A_sum = torch.sum(A_exp, dim, keepdim = True)
        score = A_exp/(A_sum + (A_sum == 0.0).float())
        '''
        #score = nn.Softmax(dim)(A)
        
        return score

    def get_local_loss(self, p_s, p_r, p_o, p_t, n_s, n_o, \
                        s_h_r, s_h_e, s_h_t, \
                        o_h_r, o_h_e, o_h_t, \
                        ):

        ent_norm = 0
        # get base intensity of pos and neg facts
        p_s_embedding = self.get_time_embedding_A(p_s, p_t.unsqueeze(1), 1)#, 2, 1)

        p_o_embedding = self.get_time_embedding_A(p_o, p_t.unsqueeze(1), 1)#, 2, 1)

        n_s_embedding = self.get_time_embedding_A(n_s, p_t.unsqueeze(1).unsqueeze(1), 2)#, 2, 2)

        n_o_embedding = self.get_time_embedding_A(n_o, p_t.unsqueeze(1).unsqueeze(1), 2)#, 2, 2)

        p_r_embedding = self.relation_embeddings(p_r) # batch_size*hidden

        '''
        p_base_score = F.dropout(p_s_embedding + p_r_embedding - p_o_embedding, p=self.prob)#batch_size
        p_base_score = torch.sum(p_base_score**2, 1).neg()
        ns_base_score = F.dropout(n_s_embedding + p_r_embedding.unsqueeze(1) - p_o_embedding.unsqueeze(1), p=self.prob)#batch_size*ns
        ns_base_score = torch.sum(ns_base_score**2, 2).neg()
        no_base_score = F.dropout(p_s_embedding.unsqueeze(1) + p_r_embedding.unsqueeze(1) - n_o_embedding, p=self.prob)
        no_base_score = torch.sum(no_base_score**2, 2).neg()
        '''
        '''
        p_base_emb = F.dropout(p_s_embedding * p_r_embedding * p_o_embedding, p=self.prob)#batch_size
        p_base_score = torch.sum(p_base_emb, 1)#/torch.norm(p_base_emb, 2, 1)
        ns_base_emb = F.dropout(n_s_embedding * p_r_embedding.unsqueeze(1) * p_o_embedding.unsqueeze(1), p=self.prob)#batch_size*ns
        ns_base_score = torch.sum(ns_base_emb, 2)#/torch.norm(ns_base_emb, 2, 2)
        no_base_emb = F.dropout(p_s_embedding.unsqueeze(1) * p_r_embedding.unsqueeze(1) * n_o_embedding, p=self.prob)
        no_base_score = torch.sum(no_base_emb, 2)#/torch.norm(no_base_emb, 2, 2)
        '''
        '''
        p_base_score = (self.Transfer_SH(p_s_embedding + p_r_embedding)) #batch_size
        p_base_score = torch.sum(p_o_embedding*p_base_score, 1)#.neg()
        ns_base_score = (self.Transfer_SH(n_s_embedding + p_r_embedding.unsqueeze(1))) #batch_size*ns
        ns_base_score = torch.sum(p_o_embedding.unsqueeze(1)*ns_base_score, 2)#.neg()
        no_base_score = (self.Transfer_SH(p_s_embedding + p_r_embedding))
        no_base_score = torch.sum(n_o_embedding*no_base_score.unsqueeze(1), 2)#.neg()
        '''
        p_base_score = self.ConvE(p_s_embedding.unsqueeze(1), p_r_embedding.unsqueeze(1), p_o_embedding.unsqueeze(1))
        ns_base_score = self.ConvE(n_s_embedding, p_r_embedding.unsqueeze(1), p_o_embedding.unsqueeze(1))
        no_base_score = self.ConvE(p_s_embedding.unsqueeze(1), p_r_embedding.unsqueeze(1), n_o_embedding)

        # get history of head entity
        s_h_r_embedding = self.relation_embeddings(s_h_r) #batch_size*time_window*relation_num(sample)*hidden
        s_h_e_embedding = self.get_time_embedding_A(s_h_e, s_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        ent_norm += Loss.normLoss(s_h_e_embedding, 4)
        o_his_embedding = self.get_time_embedding_B(p_o, s_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        ent_norm += Loss.normLoss(o_his_embedding, 4)
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - s_h_t)) #batch_size*time_window

        s_score = torch.sum((s_h_e_embedding - o_his_embedding)**2, 4).neg() + torch.sum(o_his_embedding**2, 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        #s_score = torch.sum(self.Transfer_E(s_h_e_embedding)*o_his_embedding, 4)
        s_att = self.masked_softmax(s_score, s_h_e, self.numOfEntity, 3) #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        #print(s_att[0][0][0])
        s_his_score = torch.sum(s_score*s_att, 3) # + torch.mean(torch.norm(o_his_embedding, self.norm, 4), 3) # + (s_h_r == self.numOfRelation).float() #batch_size*time_window*relation_sum(sample)

        r_score_s = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*s_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        r_att_s = self.masked_softmax(r_score_s, s_h_r, self.numOfRelation, 2) #batch_size*time_window*relation_sum(sample)
        s_his_score = torch.sum(s_his_score*r_att_s*r_score_s, 2) # + 1#*(time_decay == 1).float() #batch_size*time_window

        s_his_score = torch.sum(s_his_score*time_decay, 1) #batch_size
        #s_his_score = s_his_score + 0.5*(s_his_score == 0.0).float()


        # get history of tail entity
        o_h_r_embedding = self.relation_embeddings(o_h_r) #batch_size*time_window*relation_num(sample)*hidden
        o_h_e_embedding = self.get_time_embedding_A(o_h_e, o_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        ent_norm += Loss.normLoss(o_h_e_embedding, 4)
        s_his_embedding = self.get_time_embedding_B(p_s, o_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        ent_norm += Loss.normLoss(s_his_embedding, 4)
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - o_h_t)) #batch_size*time_window

        o_score = torch.sum((o_h_e_embedding - s_his_embedding)**2, 4).neg() + torch.sum(s_his_embedding**2, 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        #o_score = torch.sum(self.Transfer_E(o_h_e_embedding)*s_his_embedding, 4)
        o_att = self.masked_softmax(o_score, o_h_e, self.numOfEntity, 3) #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        o_his_score = torch.sum(o_score*o_att, 3) # + torch.mean(torch.norm(s_his_embedding, self.norm, 4), 3) # + (o_h_r == self.numOfRelation).float()  #batch_size*time_window*relation_sum(sample)

        r_score_o = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*o_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        r_att_o = self.masked_softmax(r_score_o, o_h_r, self.numOfRelation, 2) #batch_size*time_window*relation_sum(sample)
        o_his_score = torch.sum(o_his_score*r_att_o*r_score_o, 2)# + 1#*(time_decay == 1).float() #batch_size*time_window

        o_his_score = torch.sum(o_his_score*time_decay, 1) #batch_size
        #o_his_score = o_his_score + 0.5*(o_his_score == 0.0).float()


        # get history of head entity (with negative tail entity)
        #s_h_r_embedding = self.relation_embeddings(s_h_r) #batch_size*time_window*relation_num(sample)*hidden
        #s_h_e_embedding = self.get_time_embedding_A(s_h_e, s_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)*entity_num(sample)*hidden
        #ent_norm += Loss.normLoss(s_h_e_embedding, 4)
        no_his_embedding = self.get_time_embedding_C(n_o, s_h_t.unsqueeze(1).unsqueeze(3).unsqueeze(3).unsqueeze(3), 5) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        #ent_norm += Loss.normLoss(no_his_embedding, 5)
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - s_h_t)).unsqueeze(1) #batch_size*1*time_window

        s_score = torch.sum((s_h_e_embedding.unsqueeze(1) - no_his_embedding)**2, 5).neg() + torch.sum(no_his_embedding**2, 5) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        #s_score = torch.sum(self.Transfer_E(s_h_e_embedding).unsqueeze(1)*no_his_embedding, 5)
        #s_att = self.masked_softmax(s_score, s_h_e.unsqueeze(1), self.numOfEntity, 4) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        s_neg_score = torch.sum(s_score*s_att.unsqueeze(1), 4) # + torch.mean(torch.norm(o_his_embedding, self.norm, 4), 3) # + (s_h_r == self.numOfRelation).float() #batch_size*ns*time_window*relation_sum(sample)

        #r_score = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*s_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        #r_att = self.masked_softmax(r_score, s_h_r, self.numOfRelation, 2).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)
        s_neg_score = torch.sum(s_neg_score*r_att_s.unsqueeze(1)*r_score_s.unsqueeze(1), 3)# + 1#*(time_decay == 1).float() #batch_size*time_window

        s_neg_score = torch.sum(s_neg_score*time_decay, 2) #batch_size*ns
        #s_neg_score = s_neg_score + 0.5*(s_neg_score == 0.0).float()


        # get history of tail entity (with negative head entity)
        #o_h_r_embedding = self.relation_embeddings(o_h_r) #batch_size*time_window*relation_num(sample)*hidden
        #o_h_e_embedding = self.get_time_embedding_A(o_h_e, o_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)*entity_num(sample)*hidden
        #ent_norm += Loss.normLoss(o_h_e_embedding, 4)
        ns_his_embedding = self.get_time_embedding_C(n_s, o_h_t.unsqueeze(1).unsqueeze(3).unsqueeze(3).unsqueeze(3), 5) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        #ent_norm += Loss.normLoss(ns_his_embedding, 5)
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - o_h_t)).unsqueeze(1) #batch_size*1*time_window

        o_score = torch.sum((o_h_e_embedding.unsqueeze(1) - ns_his_embedding)**2, 5).neg() + torch.sum(ns_his_embedding**2, 5) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        #o_score = torch.sum(self.Transfer_E(o_h_e_embedding).unsqueeze(1)*ns_his_embedding, 5)
        #o_att = self.masked_softmax(o_score, o_h_e.unsqueeze(1), self.numOfEntity, 4) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        o_neg_score = torch.sum(o_score*o_att.unsqueeze(1), 4) # + torch.mean(torch.norm(o_his_embedding, self.norm, 4), 3) # + (s_h_r == self.numOfRelation).float() #batch_size*ns*time_window*relation_sum(sample)

        #r_score = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*o_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        #r_att = self.masked_softmax(r_score, o_h_r, self.numOfRelation, 2).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)
        o_neg_score = torch.sum(o_neg_score*r_att_o.unsqueeze(1)*r_score_o.unsqueeze(1), 3)# + 1#*(time_decay == 1).float() #batch_size*time_window

        o_neg_score = torch.sum(o_neg_score*time_decay, 2) #batch_size*ns
        #o_neg_score = o_neg_score + 0.5*(o_neg_score == 0.0).float()

        
        #get final score of pos and neg facts
        #print(p_base_score[0])
        #print(s_his_score[0])
        #print(o_his_score[0])

        p_s_score = p_base_score + (self.trade_off)*(s_his_score)
        p_o_score = p_base_score + (self.trade_off)*(o_his_score)
        ns_score = ns_base_score + (self.trade_off)*(o_neg_score)
        no_score = no_base_score + (self.trade_off)*(s_neg_score)

        
        #get loss
        local_loss = - torch.log(p_s_score.sigmoid() + 1e-6) \
                       - torch.log(p_o_score.sigmoid() + 1e-6) \
                       - torch.log(ns_score.neg().sigmoid() + 1e-6).sum(dim = 1) \
                       - torch.log(no_score.neg().sigmoid() + 1e-6).sum(dim = 1) \

        local_loss = local_loss.mean()
        
        return local_loss# + 0.1*ent_norm  # + 0.1*norm_loss


    def get_global_loss(self, p_s, p_o, p_t, n_s, n_o, s_h_e_flat, o_h_e_flat, p_s_d, p_o_d, p_t_m, p_s_o_r):
        p_s_embedding = self.get_time_embedding_A(p_s, p_t.unsqueeze(1), 1) #batch_size*hidden
        p_o_embedding = self.get_time_embedding_A(p_o, p_t.unsqueeze(1), 1) #batch_size*hidden

        n_s_embedding = self.get_time_embedding_A(n_s, p_t.unsqueeze(1).unsqueeze(1), 2)
        n_o_embedding = self.get_time_embedding_A(n_o, p_t.unsqueeze(1).unsqueeze(1), 2)

        p_s_o_r_embedding = self.relation_embeddings(p_s_o_r) #batch_size*sample*hidden
        r_score = torch.exp(torch.sum(p_s_o_r_embedding * self.score_vector.unsqueeze(1), 2))
        modularity = torch.sum(r_score, 1) - (p_s_d*p_o_d)/(2*p_t_m+1e-6) #batch_size
        
        s_h_e_embedding = self.get_time_embedding_A(s_h_e_flat, torch.relu(p_t - 1).unsqueeze(1).unsqueeze(1), 2) #batch_size*(time_window*sample*sample)*hidden
        #s_h_e_community = torch.sigmoid(self.auxiliary_matrix(s_h_e_embedding))#, dim=2) #batch_size*(time_window*sample*sample)*community_num
        o_h_e_embedding = self.get_time_embedding_A(o_h_e_flat, torch.relu(p_t - 1).unsqueeze(1).unsqueeze(1), 2) #batch_size*(time_window*sample*sample)*hidden
        #o_h_e_community = torch.sigmoid(self.auxiliary_matrix(o_h_e_embedding))#, dim=2) #batch_size*(time_window*sample*sample)*community_num
        
        
        p_s_his_embedding_1 = self.get_time_embedding_A(p_s, torch.relu(p_t - 1).unsqueeze(1), 1)#.unsqueeze(1) #batch_size*1*hidden
        #s_his_community = torch.sigmoid(self.auxiliary_matrix(p_s_his_embedding)) #, dim=2) #batch_size*1*community_num
        p_o_his_embedding_1 = self.get_time_embedding_A(p_o, torch.relu(p_t - 1).unsqueeze(1), 1)#.unsqueeze(1) #batch_size*1*hidden
        #o_his_community = torch.sigmoid(self.auxiliary_matrix(p_o_his_embedding))#, 2, 2) #, dim=2) #batch_size*1*community_num

        p_s_his_embedding_2 = self.get_time_embedding_A(p_s, torch.relu(p_t - 2).unsqueeze(1), 1)#.unsqueeze(1) #batch_size*1*hidden
        #s_his_community = torch.sigmoid(self.auxiliary_matrix(p_s_his_embedding)) #, dim=2) #batch_size*1*community_num
        p_o_his_embedding_2 = self.get_time_embedding_A(p_o, torch.relu(p_t - 2).unsqueeze(1), 1)#.unsqueeze(1) #batch_size*1*hidden
        #o_his_community = torch.sigmoid(self.auxiliary_matrix(p_o_his_embedding))#, 2, 2) #, dim=2) #batch_size*1*community_num

        p_s_his_embedding_3 = self.get_time_embedding_A(p_s, torch.relu(p_t - 3).unsqueeze(1), 1)#.unsqueeze(1) #batch_size*1*hidden
        #s_his_community = torch.sigmoid(self.auxiliary_matrix(p_s_his_embedding)) #, dim=2) #batch_size*1*community_num
        p_o_his_embedding_3 = self.get_time_embedding_A(p_o, torch.relu(p_t - 3).unsqueeze(1), 1)#.unsqueeze(1) #batch_size*1*hidden
        #o_his_community = torch.sigmoid(self.auxiliary_matrix(p_o_his_embedding))#, 2, 2) #, dim=2) #batch_size*1*community_num

        '''
        sample = rd.sample(range(0,self.numOfEntity), 300)
        all_his_embedding = self.get_time_embedding_A(torch.LongTensor(sample).unsqueeze(0).cuda(), torch.relu(p_t - 1).unsqueeze(1).unsqueeze(1), 2) #batch_size*sample_num*hidden
        all_his_community = torch.sigmoid(self.auxiliary_matrix(all_his_embedding)) #, 2, 2)
        '''
        
        '''
        s_mask = ((all_his_community >= 0.2).float() * (s_his_community >= 0.2).float()).sum(2)
        s_mask = (s_mask != 0).float()
        s_C_embedding = torch.mean(s_com_embedding*s_mask.unsqueeze(2) , 1)

        o_mask = ((all_his_community >= 0.2).float() * (o_his_community >= 0.2).float()).sum(2)
        o_mask = (o_mask != 0).float()
        o_C_embedding = torch.mean(o_com_embedding*o_mask.unsqueeze(2) , 1)
        '''

        p_s_community = self.auxiliary_matrix(p_s_embedding + p_s_his_embedding_1 + p_s_his_embedding_2 + p_s_his_embedding_3).sigmoid() #, dim=1) #batch_size*community_num
        #p_s_community = p_s_embedding
        #p_s_community = ((torch.abs(p_s_community) + 1e-6)**0.5)/torch.sum((torch.abs(p_s_community) + 1e-6)**0.5, 1, keepdim = True)*(-1)*(p_s_community < 0).float()
        F.normalize(p_s_community, 2, 1)

        p_o_community = self.auxiliary_matrix(p_o_embedding + p_o_his_embedding_1 + p_o_his_embedding_2 + p_o_his_embedding_3).sigmoid() #, dim=1) #batch_size*community_num
        #p_o_community = p_o_embedding
        #p_o_community = ((torch.abs(p_o_community) + 1e-6)**0.5)/torch.sum((torch.abs(p_o_community) + 1e-6)**0.5, 1, keepdim = True)*(-1)*(p_o_community < 0).float()
        F.normalize(p_o_community, 2, 1)

        n_s_community = self.auxiliary_matrix(n_s_embedding).sigmoid() #, dim=2)
        #n_s_community = n_s_embedding
        #n_s_community = ((torch.abs(n_s_community) + 1e-6)**0.5)/torch.sum((torch.abs(n_s_community) + 1e-6)**0.5, 2, keepdim = True)*(-1)*(n_s_community < 0).float()
        F.normalize(n_s_community, 2, 2)

        n_o_community = self.auxiliary_matrix(n_o_embedding).sigmoid() #, dim=2)
        #n_o_community = n_o_embedding
        #n_o_community = ((torch.abs(n_o_community) + 1e-6)**0.5)/torch.sum((torch.abs(n_o_community) + 1e-6)**0.5, 2, keepdim = True)*(-1)*(n_o_community < 0).float()
        F.normalize(n_o_community, 2, 2)

        p_s_score = torch.sum(p_s_community*p_o_community, 1)#*modularity
        n_s_score = torch.sum(n_s_community*p_o_community.unsqueeze(1), 2).neg()
        n_o_score = torch.sum(p_s_community.unsqueeze(1)*n_o_community, 2).neg()
        
        global_loss = - p_s_score #\
                      #- n_s_score.mean(dim = 1) \
                      #- n_o_score.mean(dim = 1) \
        
        '''
        p_s_score = torch.sum(p_s_community*p_o_community, 1)
        n_s_score = torch.sum((p_s_community.unsqueeze(1)*n_o_community), 2)*modularity.unsqueeze(1)
        n_o_score = torch.sum((p_o_community.unsqueeze(1)*n_s_community), 2)*modularity.unsqueeze(1)
        
        global_loss = - p_s_score \
                      - n_s_score.sum(dim = 1) \
                      - n_o_score.sum(dim = 1) \
        '''

        global_loss = global_loss.mean()
        norm_loss = 0
        for W in self.auxiliary_matrix.parameters():
            norm_loss += W.norm(2)

        return global_loss + 0.01*norm_loss

    def get_co_loss(self):

        return 0


    def validate(self,p_s, p_r, p_o, p_t, \
                s_h_r, s_h_e, s_h_t, \
                o_h_r, o_h_e, o_h_t, \
                TrainData):
        MRR = 0.0
        H1 = 0.0
        H3 = 0.0
        H5 = 0.0
        H10 = 0.0
        # get base intensity of candidate facts
        p_s_embedding = self.get_time_embedding_A(p_s, p_t.unsqueeze(1), 1)# batch_size*hidden
        p_o_embedding = self.get_time_embedding_A(p_o, p_t.unsqueeze(1), 1)
        all_embedding = self.get_time_embedding_D(p_t.unsqueeze(1).unsqueeze(1), 2)# batch_size*numOfEntity*hidden

        p_r_embedding = self.relation_embeddings(p_r)
        '''
        base_score_target = torch.sum((p_s_embedding + p_r_embedding - p_o_embedding)**2, 1).unsqueeze(1).neg() # batch_size*1
        base_score_head = torch.sum((all_embedding + p_r_embedding.unsqueeze(1) - p_o_embedding.unsqueeze(1))**2, 2).neg() 
        base_score_tail = torch.sum((p_s_embedding.unsqueeze(1) + p_r_embedding.unsqueeze(1) - all_embedding)**2, 2).neg() # batch_size*numOfEntity
        '''
        '''
        p_base_emb = p_s_embedding * p_r_embedding * p_o_embedding
        base_score_target = torch.sum(p_base_emb, 1)#/torch.norm(p_base_emb, 2, 1)
        ns_base_emb = all_embedding * p_r_embedding.unsqueeze(1) * p_o_embedding.unsqueeze(1)#batch_size*ns
        base_score_head = torch.sum(ns_base_emb, 2)#/torch.norm(ns_base_emb, 2, 2)
        no_base_emb = p_s_embedding.unsqueeze(1) * p_r_embedding.unsqueeze(1) * all_embedding
        base_score_tail = torch.sum(no_base_emb, 2)#/torch.norm(no_base_emb, 2, 2)
        '''
        '''
        base_score_target = (self.Transfer_SH(p_s_embedding + p_r_embedding)) #batch_size
        base_score_target = torch.sum(p_o_embedding*base_score_target, 1)#.neg()
        base_score_head = (self.Transfer_SH(all_embedding + p_r_embedding.unsqueeze(1))) #batch_size*ns
        base_score_head = torch.sum(p_o_embedding.unsqueeze(1)*base_score_head, 2)#.neg()
        base_score_tail = (self.Transfer_SH(p_s_embedding + p_r_embedding))
        base_score_tail = torch.sum(all_embedding*base_score_tail.unsqueeze(1), 2)#.neg()
        '''
        base_score_target = self.ConvE(p_s_embedding.unsqueeze(1), p_r_embedding.unsqueeze(1), p_o_embedding.unsqueeze(1))
        base_score_head = self.ConvE(all_embedding, p_r_embedding.unsqueeze(1), p_o_embedding.unsqueeze(1))
        base_score_tail = self.ConvE(p_s_embedding.unsqueeze(1), p_r_embedding.unsqueeze(1), all_embedding)
        # get history of head entity
        
        s_h_r_embedding = self.relation_embeddings(s_h_r) #batch_size*time_window*relation_num(sample)*hidden
        s_h_e_embedding = self.get_time_embedding_A(s_h_e, s_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        o_his_embedding = self.get_time_embedding_B(p_o, s_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - s_h_t)) #batch_size*time_window

        s_score = torch.sum((s_h_e_embedding - o_his_embedding)**2, 4).neg() + torch.sum(o_his_embedding**2, 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        #s_score = torch.sum(self.Transfer_E(s_h_e_embedding)*o_his_embedding, 4)
        s_att = self.masked_softmax(s_score, s_h_e, self.numOfEntity, 3) #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        s_his_score = torch.sum(s_score*s_att, 3)# + torch.mean(torch.norm(o_his_embedding, self.norm, 4), 3) # + (s_h_r == self.numOfRelation).float() #batch_size*time_window*relation_sum(sample)

        r_score_s = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*s_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        r_att_s = self.masked_softmax(r_score_s, s_h_r, self.numOfRelation, 2) #batch_size*time_window*relation_sum(sample)
        s_his_score = torch.sum(s_his_score*r_att_s*r_score_s, 2)# + 1#*(time_decay == 1).float() #batch_size*time_window

        s_his_score = torch.sum(s_his_score*time_decay, 1) #batch_size
        #s_his_score = s_his_score + 0.5*(s_his_score == 0.0).float()


        # get history of tail entity
        o_h_r_embedding = self.relation_embeddings(o_h_r) #batch_size*time_window*relation_num(sample)*hidden
        o_h_e_embedding = self.get_time_embedding_A(o_h_e, o_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        s_his_embedding = self.get_time_embedding_B(p_s, o_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - o_h_t)) #batch_size*time_window

        o_score = torch.sum((o_h_e_embedding - s_his_embedding)**2, 4).neg() + torch.sum(s_his_embedding**2, 4) #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        #o_score = torch.sum(self.Transfer_E(o_h_e_embedding)*s_his_embedding, 4)
        o_att = self.masked_softmax(o_score, o_h_e, self.numOfEntity, 3)  #batch_size*time_window*relation_sum(sample)*entity_num(sample)
        o_his_score = torch.sum(o_score*o_att, 3)# + torch.mean(torch.norm(s_his_embedding, self.norm, 4), 3) # + (o_h_r == self.numOfRelation).float() #batch_size*time_window*relation_sum(sample)

        r_score_o = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*o_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        r_att_o = self.masked_softmax(r_score_o, o_h_r, self.numOfRelation, 2) #batch_size*time_window*relation_sum(sample)
        o_his_score = torch.sum(o_his_score*r_att_o*r_score_o, 2)# + 1#*(time_decay == 1).float() #batch_size*time_window

        o_his_score = torch.sum(o_his_score*time_decay, 1) #batch_size
        #o_his_score = o_his_score + 0.5*(o_his_score == 0.0).float()
        
        # get history of head entity (with negative tail entity)
        #s_h_r_embedding = self.relation_embeddings(s_h_r) #batch_size*time_window*relation_num(sample)*hidden
        #s_h_e_embedding = self.get_time_embedding_A(s_h_e, s_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)*entity_num(sample)*hidden
        no_his_embedding = self.get_time_embedding_E(s_h_t.unsqueeze(1).unsqueeze(3).unsqueeze(3).unsqueeze(3), 5) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - s_h_t)).unsqueeze(1) #batch_size*1*time_window

        s_score = torch.sum((s_h_e_embedding.unsqueeze(1) - no_his_embedding)**2, 5).neg() + torch.sum(no_his_embedding**2, 5) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        #s_score =  torch.sum(self.Transfer_E(s_h_e_embedding.unsqueeze(1))*no_his_embedding, 5)
        #s_att = self.masked_softmax(s_score, s_h_e.unsqueeze(1), self.numOfEntity, 4) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        s_neg_score = torch.sum(s_score*s_att.unsqueeze(1), 4) # + torch.mean(torch.norm(o_his_embedding, self.norm, 4), 3) # + (s_h_r == self.numOfRelation).float() #batch_size*ns*time_window*relation_sum(sample)

        #r_score = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*s_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        #r_att = self.masked_softmax(r_score, s_h_r, self.numOfRelation, 2).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)
        s_neg_score = torch.sum(s_neg_score*r_att_s.unsqueeze(1)*r_score_s.unsqueeze(1), 3)# + 1#*(time_decay == 1).float() #batch_size*time_window

        s_neg_score = torch.sum(s_neg_score*time_decay, 2) #batch_size*ns
        #s_neg_score = s_neg_score + 0.5*(s_neg_score == 0.0).float()



        # get history of tail entity (with negative head entity)
        #o_h_r_embedding = self.relation_embeddings(o_h_r) #batch_size*time_window*relation_num(sample)*hidden
        #o_h_e_embedding = self.get_time_embedding_A(o_h_e, o_h_t.unsqueeze(2).unsqueeze(2).unsqueeze(2), 4).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)*entity_num(sample)*hidden
        ns_his_embedding = self.get_time_embedding_E(o_h_t.unsqueeze(1).unsqueeze(3).unsqueeze(3).unsqueeze(3), 5) #batch_size*time_window*relation_sum(sample)*entity_num(sample)*hidden
        time_decay = torch.exp(-torch.abs(self.decay_rate)*torch.abs(p_t.unsqueeze(1) - o_h_t)).unsqueeze(1) #batch_size*1*time_window

        o_score = torch.sum((o_h_e_embedding.unsqueeze(1) - ns_his_embedding)**2, 5).neg() + torch.sum(ns_his_embedding**2, 5) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        #o_score = torch.sum(self.Transfer_E(o_h_e_embedding).unsqueeze(1)*ns_his_embedding, 5)
        #o_att = self.masked_softmax(o_score, o_h_e.unsqueeze(1), self.numOfEntity, 4) #batch_size*ns*time_window*relation_sum(sample)*entity_num(sample)
        o_neg_score = torch.sum(o_score*o_att.unsqueeze(1), 4) # + torch.mean(torch.norm(o_his_embedding, self.norm, 4), 3) # + (s_h_r == self.numOfRelation).float() #batch_size*ns*time_window*relation_sum(sample)

        #r_score = torch.sum(self.Transfer_R(p_r_embedding).unsqueeze(1).unsqueeze(1)*o_h_r_embedding, 3) #batch_size*time_window*relation_sum(sample)
        #r_att = self.masked_softmax(r_score, o_h_r, self.numOfRelation, 2).unsqueeze(1) #batch_size*1*time_window*relation_sum(sample)
        o_neg_score = torch.sum(o_neg_score*r_att_o.unsqueeze(1)*r_score_o.unsqueeze(1), 3)# + 1#*(time_decay == 1).float() #batch_size*time_window

        o_neg_score = torch.sum(o_neg_score*time_decay, 2) #batch_size*ns
        #o_neg_score = o_neg_score + 0.5*(o_neg_score == 0.0).float()
        #print(o_neg_score.size())
        
        for i in range(len(p_s)):
            s_i = int(p_s[i])
            r_i = int(p_r[i])
            o_i = int(p_o[i])
            t_i = int(p_t[i])

            # get final score pos and neg fact
            #target_s_score = self.trade_off*base_score_target[i] + (1 - self.trade_off)*(s_his_score[i])
            #target_o_score = self.trade_off*base_score_target[i] + (1 - self.trade_off)*(o_his_score[i])
            tmp_head_score = (base_score_head[i] + (self.trade_off)*(o_neg_score[i])).squeeze(0)
            tmp_tail_score = (base_score_tail[i] + (self.trade_off)*(s_neg_score[i])).squeeze(0)

            # get rank
            tmp_head = (- tmp_head_score[s_i] + tmp_head_score)
            tmp_tail = (- tmp_tail_score[o_i] + tmp_tail_score)
            '''
            if s_i in TrainData["er2e"].keys() and r_i in TrainData["er2e"][s_i].keys() and t_i in TrainData["er2e"][s_i][r_i].keys():
                tail_list = list(set(TrainData["er2e"][s_i][r_i][t_i]))
                tmp_tail[tail_list] = 0
            if o_i in TrainData["er2e"].keys() and r_i in TrainData["er2e"][o_i].keys() and t_i in TrainData["er2e"][o_i][r_i].keys():
                head_list = list(set(TrainData["er2e"][o_i][r_i][t_i]))
                tmp_head[head_list] = 0
            '''
            wrongHead=torch.nonzero(nn.functional.relu(tmp_head))
            wrongTail=torch.nonzero(nn.functional.relu(tmp_tail))

            Rank_H=wrongHead.size()[0]+1
            Rank_T=wrongTail.size()[0]+1

            MRR += 1/Rank_H+ 1/Rank_T
            
            if Rank_H<=1:
                H1+=1
            if Rank_T<=1:
                H1+=1

            if Rank_H<=3:
                H3+=1
            if Rank_T<=3:
                H3+=1

            if Rank_H<=5:
                H5+=1
            if Rank_T<=5:
                H5+=1

            if Rank_H<=10:
                H10+=1
            if Rank_T<=10:
                H10+=1

        return MRR, H1, H3, H5, H10

class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(out_channels*input_dim, 1,bias = False)

        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, s, r, o):

        conv_input = torch.cat([s, r, o], 1)
        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        #conv_input = self.conv1_bn(conv_input)
        #print(conv_input.size())
        out_conv = self.conv_layer(conv_input)
        #out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)

        out_conv = self.dropout(out_conv)

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        #print(input_fc.size())
        #input_fc = s+r-o
        output = self.fc_layer(input_fc.view(-1, 500))
        #output = torch.norm(input_fc, 1, 1)
        return output