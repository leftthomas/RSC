# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss


class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0, 2, 1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):
        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA']
        )

        loss = HA_refinement + HB_refinement
        return loss


def local_response_loss(rgb, flow, num_block):
    n, d, l = rgb.shape
    # [N, B, L/B, D]
    rgb = rgb.transpose(-1, -2).reshape(n, num_block, -1, d)
    flow = flow.transpose(-1, -2).reshape(n, num_block, -1, d)
    # [N, B, D]
    rgb, flow = torch.mean(rgb, dim=-2), torch.mean(flow, dim=-2)
    rgb, flow = F.normalize(rgb, dim=-1), F.normalize(flow, dim=-1)
    # [N, B, B]
    rgb_sim = torch.matmul(rgb, rgb.transpose(-1, -2))
    flow_sim = torch.matmul(flow, flow.transpose(-1, -2))
    return F.mse_loss(rgb_sim, flow_sim)


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)

        # local response consistency loss
        loss_lrc = local_response_loss(contrast_pairs['RGB'], contrast_pairs['FLOW'], contrast_pairs['NUM_BLOCK'])

        loss_total = loss_cls + 0.005 * loss_snico + loss_lrc

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/LRC': loss_lrc,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict
