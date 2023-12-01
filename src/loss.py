import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Union, Iterable, Dict


def bdcn_loss2(inputs: Tensor, targets: Tensor, l_weight=1.1, reduction = None):
    # bdcn loss modified in DexiNed

    # targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.5).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.5).float()).float() # <= 0.1

    mask[mask > 0.5] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.5] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # inputs= torch.sigmoid(inputs)
    # cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    cost = F.binary_cross_entropy(input= inputs, target= targets.float(), weight= mask, reduction= 'none')
    if reduction is not None:
        ops = getattr(torch, reduction)
    else:
        ops = lambda x: x
    cost = ops(cost) # before sum
    return l_weight*cost


def bdrloss(prediction: Tensor, label: Tensor, radius, reduction):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(prediction.device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)



    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0
    if reduction is not None:
        ops = getattr(torch, reduction)
    else:
        ops = lambda x: x
    return ops(cost)



def textureloss(prediction, label, mask_radius, reduction):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
        
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(prediction.device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(prediction.device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    if reduction is not None:
        ops = getattr(torch, reduction)
    else:
        ops = lambda x: x
    return ops(loss)


def tracingloss(prediction, label, tex_factor=0., bdr_factor=0., bdcn_factor=0.,  balanced_w=1.1, reduction: Union[None, str] = 'mean'):
    label = label.float()
    prediction = prediction.float()

    # if not torch.all(torch.ge(prediction, 0) * torch.le(prediction, 1)):
    #     prediction = torch.sigmoid(prediction)
    with torch.no_grad():
        mask = label.clone()

        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    
    if reduction is not None:
        ops = getattr(torch, reduction)
    else:
        ops = lambda x: x
    cost = ops(F.binary_cross_entropy(
                prediction.float(),label.float(), weight=mask, reduce=False))
    label_w = (label != 0).float()


    textcost = textureloss(prediction.float(),label_w.float(), mask_radius=2, reduction = reduction)

    bdrcost = bdrloss(prediction.float(),label_w.float(), radius=2, reduction = reduction)

    # bdcncost = bdcn_loss2(prediction.float(),label_w.float(), l_weight=1.1, reduction = reduction)

    return cost + bdr_factor*bdrcost + tex_factor*textcost


class TracingLoss(torch.nn.Module):
    def __init__(self, tex_factor=0., bdr_factor=0., bdcn_factor = 0.2, balanced_w=1.1, reduction = "mean") -> None:
        super().__init__()
        assert reduction in ["mean", "sum", "none", None]
        self.tex_factor= tex_factor
        self.bdr_factor= bdr_factor
        self.bdcn_factor = bdcn_factor
        self.balanced = balanced_w
        self.reduction = reduction if reduction in ["mean", "sum"] else None

    def forward(self, predictions: Union[Dict, Iterable[Tensor], Tensor], targets: Tensor) -> Tensor:
        if isinstance(predictions, torch.Tensor):
            return tracingloss(predictions, targets, self.tex_factor, self.bdr_factor, self.bdcn_factor, self.balanced, self.reduction)
        else:
            losses = []
            try:
                if isinstance(predictions, Dict):
                    predictions = predictions.values()
            finally:
                for prediction in predictions:
                    local_prediction = F.interpolate(prediction, size= targets.shape[-2:])
                    losses.append(tracingloss(local_prediction, targets, self.tex_factor, self.bdr_factor, self.bdcn_factor, self.balanced, "mean"))
            
            return torch.stack(losses).mean()
    

