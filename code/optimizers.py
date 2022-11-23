"""Copied from: https://www.sri.inf.ethz.ch/teaching/rtai22 solutions of week2 exercises"""

import torch

def fgsm_(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""    
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits = model(input_)
    target = torch.LongTensor([target])
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    
    #perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()
    
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    
    return out

# x: input image
# label: current label of x
# k: number of FGSM iterations
# eps: size of l-infinity ball
# eps_step: step size of FGSM iterations
def pgd(model, x, target, k, eps, eps_step, targeted, clip_min, clip_max):
    # TODO: insert your implementation for PGD here 

    # create a copy of the input, remove all previous associations to the compute graph...
    x_adv = x.clone().detach_()
    # pick a random point in the eps sized box:
    x_adv = x_adv + eps * (2 * torch.rand(x_adv.size()) - 1)

    target = torch.LongTensor([target])
    CE_Loss = nn.CrossEntropyLoss()

    # k FGSM iterations:
    for _ in range(k):
        # FGSM step. We don't clamp here (arguments clip_min=None, clip_max=None) 
        # as we want to apply the attack as defined
        x_adv = fgsm_(model, x_adv, target, eps, targeted, clip_min=None, clip_max=None)
        # Project back to box of size eps
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)

    return x_adv