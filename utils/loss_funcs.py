#!/usr/bin/env python
# coding: utf-8

import torch


def mpjpe_error(batch_pred,batch_gt): 

    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))

def fde_error(batch_pred,batch_gt): 

    batch_pred=batch_pred[:,-1,:,:].contiguous().view(-1,3)
    batch_gt=batch_gt[:,-1,:,:].contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))

def weighted_mpjpe_error(batch_pred,batch_gt, joint_weights): 
    # 'BackTop', 'LShoulderBack', 'RShoulderBack',
    #                   'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut'
    batch_pred=batch_pred.contiguous()
    batch_gt=batch_gt.contiguous()
    diff = batch_gt - batch_pred
    diff *= joint_weights

    return torch.mean(torch.norm(diff.view(-1,3),2,1))

def perjoint_error(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous()
    batch_gt=batch_gt.contiguous()
    diff = batch_gt - batch_pred
    batch_joint_errors = torch.mean(torch.norm(diff,2,3), 1)
    return torch.mean(batch_joint_errors,0)

def perjoint_error_eval(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous()
    batch_gt=batch_gt.contiguous()
    diff = batch_gt - batch_pred
    batch_joint_errors = torch.mean(torch.norm(diff,2,3), 1)
    return torch.mean(batch_joint_errors,0), batch_joint_errors

def perjoint_fde(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous()
    batch_gt=batch_gt.contiguous()
    diff = batch_gt - batch_pred
    batch_joint_errors = torch.norm(diff[:, -1],2,2)
    return torch.mean(batch_joint_errors,0), batch_joint_errors




