# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:04:41 2020

@author: zjs
"""
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        #loss = -(target*input.log() + (1-target)*(1-input).log())
        return loss.mean()

class sigmoidLoss(nn.Module):
    def __init__(self):
        super(sigmoidLoss, self).__init__()

    def forward(self, pos, neg):
        pos_score = -F.logsigmoid(pos).mean()
        neg_score = -F.logsigmoid(-neg).mean()
        return (pos_score+neg_score)/2

class marginLoss(nn.Module):
	def __init__(self):
		super(marginLoss, self).__init__()

	def forward(self, pos, neg, margin):
		zero_tensor = floatTensor(pos.size())
		zero_tensor.zero_()
		zero_tensor = autograd.Variable(zero_tensor)
		return torch.sum(torch.max(pos - neg + margin, zero_tensor))
    
    
class double_marginLoss(nn.Module):
    def __init__(self):
        super(double_marginLoss,self).__init__()
        
    def forward(self, pos, neg, margin):
        zero_tensor = floatTensor(pos.size())
        zero_tensor.zero_()
        zero_tensor = autograd.Variable(zero_tensor)
        
        pos_margin=1.0
        neg_margin=pos_margin+margin
        return torch.sum(torch.max(neg_margin-neg, zero_tensor))+torch.sum(torch.max(pos-pos_margin,zero_tensor))

def orthogonalLoss(rel_embeddings, norm_embeddings):
	return torch.sum(torch.sum(torch.mm(rel_embeddings,norm_embeddings), dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def normLoss(embeddings, dim=1):
	norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
	return torch.mean(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))
def F_norm(martix):
    norm = torch.norm(martix)
    return torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0])))
