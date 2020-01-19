"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.autograd


def make_one_hot(labels, C=10):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    """
    target = torch.eye(C)[labels.data]
    target = target.to(labels.get_device())
    return target

class CWLoss(nn.Module):
    def __init__(self):
        super(CWLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, logits, labels, C=10):
        label_mask = make_one_hot(labels, C)
        wrong_logit, _ = torch.max((1-label_mask)*logits - 1e4*label_mask, dim=1)
        correct_logit = torch.sum(logits * label_mask, dim=1)
        loss = -self.relu(correct_logit - wrong_logit + 50)
        return loss

class MRKLD(nn.Module):
    def __init__(self):
        super(MRKLD, self).__init__()

    def forward(self, logits, labels):
        log_softmax = self.log_softmax(logits)
        E = torch.sum(log_softmax, dim=1)
        return E

class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, loss_func, device=0):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.device = device

        if loss_func == 'xent':
            loss = nn.CrossEntropyLoss()
        elif loss_func == 'cw':
            loss = CWLoss()
        elif loss_func == 'mrkld':
            loss = MRKLD()
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = nn.CrossEntropyLoss()

        self.loss = loss

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + torch.FloatTensor(*x_nat.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
        else:
            x = x_nat
        x.requires_grad = True
        for i in range(self.k):
            output = self.model(x)
            loss = self.loss(output, y)
            grad_outs = torch.ones_like(output)
            grad = torch.autograd.grad(loss, x, grad_outputs=grad_outs,
                      retain_graph=True, create_graph=True,
                      allow_unused=True)[0]

            x += self.a * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
        return x