# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


# Multi-head Cross Modal Attention
class CMA(nn.Module):
    def __init__(self, feat_dim, num_head):
        super(CMA, self).__init__()
        self.rgb_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.flow_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.atte = nn.Parameter(torch.empty(num_head, feat_dim // num_head, feat_dim // num_head))

        nn.init.uniform_(self.rgb_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.flow_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.atte, -math.sqrt(feat_dim // num_head), math.sqrt(feat_dim // num_head))
        self.num_head = num_head

    def forward(self, rgb, flow):
        n, l, d = rgb.shape
        # [N, H, L, D/H]
        o_rgb = F.normalize(torch.matmul(rgb.unsqueeze(dim=1), self.rgb_proj), dim=-1)
        o_flow = F.normalize(torch.matmul(flow.unsqueeze(dim=1), self.flow_proj), dim=-1)
        # [N, H, L, L]
        atte = torch.matmul(torch.matmul(o_rgb, self.atte), o_flow.transpose(-1, -2))
        rgb_atte = torch.softmax(atte, dim=-1)
        flow_atte = torch.softmax(atte.transpose(-1, -2), dim=-1)

        # [N, H, L, D/H]
        e_rgb = F.gelu(torch.matmul(rgb_atte, o_rgb))
        e_flow = F.gelu(torch.matmul(flow_atte, o_flow))
        # [N, L, D]
        f_rgb = torch.tanh(e_rgb.transpose(1, 2).reshape(n, l, -1) + rgb)
        f_flow = torch.tanh(e_flow.transpose(1, 2).reshape(n, l, -1) + flow)
        return f_rgb, f_flow


# (a) Feature Embedding and (b) Actionness Modeling
class Actionness_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Actionness_Module, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=9,
                      stride=1, padding=4),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=9,
                      stride=1, padding=4, bias=False),
            nn.Conv1d(num_classes, num_classes, kernel_size=7, stride=1, padding=6, dilation=2, bias=False,
                      groups=num_classes),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.f_embed(out)
        embeddings = out.permute(0, 2, 1)
        out = self.dropout(out)
        out = self.f_cls(out)
        cas = out.permute(0, 2, 1)
        actionness = cas.sum(dim=2)
        return embeddings, cas, actionness


# CoLA Pipeline
class CoLA(nn.Module):
    def __init__(self, cfg):
        super(CoLA, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES

        self.actionness_module = Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_easy = cfg.R_EASY
        self.r_hard = cfg.R_HARD
        self.m = cfg.m
        self.M = cfg.M

        self.dropout = nn.Dropout(p=0.6)

        self.cma = CMA(cfg.FEATS_DIM // 2, cfg.NUM_HEAD)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act = self.select_topk_embeddings(actionness_drop, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)

        return easy_act, easy_bkg

    def hard_snippets_mining(self, actionness, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _ = cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores

    def forward(self, x):
        num_segments = x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard

        rgb, flow = x[:, :, :1024], x[:, :, 1024:]
        rgb, flow = self.cma(rgb, flow)
        x = torch.cat((rgb, flow), dim=-1)

        embeddings, cas, actionness = self.actionness_module(x)

        easy_act, easy_bkg = self.easy_snippets_mining(actionness, embeddings, k_easy)
        hard_act, hard_bkg = self.hard_snippets_mining(actionness, embeddings, k_hard)

        video_scores = self.get_video_cls_scores(cas, k_easy)

        contrast_pairs = {
            'EA': easy_act,
            'EB': easy_bkg,
            'HA': hard_act,
            'HB': hard_bkg
        }

        return video_scores, contrast_pairs, actionness, cas