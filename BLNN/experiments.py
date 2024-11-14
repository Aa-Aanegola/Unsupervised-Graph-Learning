import torch
from torch_scatter.composite import scatter_softmax

def bgrl_loss(q1, q2, y1, y2, src, dst, FLAGS):
    loss_self = (
        2
        - (q1*y2).sum()/q1.shape[0]
        - (q2*y1).sum()/q2.shape[0]
    )
    return loss_self


def blnn_loss(q1, q2, y1, y2, src, dst, FLAGS):
    # Bootstrap Latents of Neighbors
    attn = (y1[src]*y2[dst]).sum(1)
    attn = scatter_softmax(attn/FLAGS.tau, dst)
    
    nei1 = (q1[src]*y2[dst]).sum(1)
    nei2 = (q2[src]*y1[dst]).sum(1)
    
    loss_neig = (
        - (attn*nei1).sum()/q1.shape[0] 
        - (attn*nei2).sum()/q2.shape[0]
    )
    
    return loss_neig