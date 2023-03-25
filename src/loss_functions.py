import torch

def ranknet_loss(pred1, pred2, target_probability):
    """
    Computes the ranknet loss between 2 consecutive predictions with respect to a target probability.
    """
   

    o12 = pred1 - pred2
    print(torch.exp(o12))
    losses =  -target_probability * o12 + torch.log(1 + torch.exp(o12))

    # print(losses)

    return torch.mean(losses)