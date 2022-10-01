from torch import nn, optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Optional
#from torch.optim.adamw import adamw
import math

class adv_optimizer(optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,limit=1,norm_type=1,
                 weight_decay=1e-2, amsgrad=False, *, maximize=False,
                 foreach = None,capturable=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        super(adv_optimizer, self).__init__(params, defaults)
        self.limit = limit
        self.norm_type = norm_type

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                #pgd process for adv parameters
                if group['lr']<0:
                    #adv parameters, update using our pgd
                    #for p in group['params']:
                    # if p.grad is None:
                    #     continue
                    if self.norm_type==1:
                        adv_norm = torch.norm(p,p=2)
                        adv_percent = min(self.limit,adv_norm)/adv_norm
                        p.mul_(adv_percent)
                    elif self.norm_type==2:
                        #adv_sign = torch.sign(p)

                        #adv_norm = torch.norm(p, float('inf'))
                        #adv_percent = min(adv_norm,self.limit)
                        p.clamp_(-self.limit,self.limit)
                        #p.mul_(adv_percent)
                       # p = adv_sign*adv_percent
                    if self.norm_type==1:
                        print("updating adv ",adv_norm,adv_percent)

                # if p.grad is None:
                #     continue
                # params_with_grad.append(p)
                # if p.grad.is_sparse:
                #     raise RuntimeError('AdamW does not support sparse gradients')
                # grads.append(p.grad)
                #
                # state = self.state[p]
                #
                # # State initialization
                # if len(state) == 0:
                #     state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                #         if self.defaults['capturable'] else torch.tensor(0.)
                #     # Exponential moving average of gradient values
                #     state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                #     # Exponential moving average of squared gradient values
                #     state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                #     if amsgrad:
                #         # Maintains max of all exp. moving avg. of sq. grad. values
                #         state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                #
                # exp_avgs.append(state['exp_avg'])
                # exp_avg_sqs.append(state['exp_avg_sq'])
                #
                # if amsgrad:
                #     max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                #
                # state_steps.append(state['step'])






        return loss


