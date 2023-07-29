import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is a wessertein distance class.
"""
class Wessertein_Div(nn.Module):
    r"""References:
        [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.
    """
    def __init__(self, p=1):
        super(Wessertein_Div, self).__init__()
        self.p = p

    def forward1(self, u_values, v_values):
        u_values = F.normalize(u_values, p=1, dim=-1)  # 归一化p分布
        v_values = F.normalize(v_values, p=1, dim=-1)  # 归一化q分布

        # 计算每个分布的累积分布函数
        p_cdf = torch.cumsum(u_values, dim=-1)
        q_cdf = torch.cumsum(v_values, dim=-1)

        # 计算Wasserstein距离
        wes_distance = torch.mean(torch.abs(p_cdf - q_cdf))
        return wes_distance

    def forward(self, u_values, v_values):
        # p.shape : (N, 2 * N), q.shape : (N, 2 * N)
        u_values = F.normalize(u_values, p=1, dim=-1)  # 归一化p分布
        v_values = F.normalize(v_values, p=1, dim=-1)  # 归一化q分布

        u_sorter = torch.argsort(u_values, dim=-1) # (N, 2 * N)
        v_sorter = torch.argsort(v_values, dim=-1) # (N, 2 * N)

        all_values = torch.cat((u_values, v_values), dim=-1) # (N, 4 * N)
        all_values, sorted_indices = torch.sort(all_values, dim=-1)

        deltas = torch.diff(all_values, dim=-1) # # (N, 4 * N)

        u_cdf = u_sorter / u_values.shape[-1]  # (N, 2 * N)
        v_cdf = v_sorter / v_values.shape[-1]  # (N, 2 * N)

        if self.p == 1:
            wes_distance = torch.mean(torch.mm(torch.abs(u_cdf - v_cdf).T, deltas))
        elif self.p == 2:
            wes_distance = torch.sqrt(torch.mean(torch.mm(torch.square(u_cdf - v_cdf).T, deltas)))
        else:
            wes_distance = torch.pow(torch.mean(torch.mm(torch.pow(torch.abs(u_cdf - v_cdf), self.p).T, deltas)), 1/self.p)
        return wes_distance

"""
This is a JS distance class.
"""
class JS_Div(nn.Module):
    def __init__(self):
        super(JS_Div, self).__init__()

    def forward(self, p, q, eps=1e-8):
        p = F.sigmoid(p)
        q = F.sigmoid(q)
        # Add epsilon for numerical stability
        p = p + eps
        q = q + eps
        
        # Normalize p and q
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute mean of p and q
        m = 0.5 * (p + q)
        
        # Compute KL divergence
        kl_p = (p * torch.log2(p / m)).sum()
        kl_q = (q * torch.log2(q / m)).sum()
        
        # Compute JS divergence
        jsd = 0.5 * (kl_p + kl_q)
        return jsd

class IALLoss(nn.Module):
    def __init__(self, tau=0.05, alpha=0.5, zoom=0.1, n_view=2, inversion=False, reduction="mean", detach=False):
        super(IALLoss, self).__init__()
        self.tau = tau
        self.alpha = alpha
        self.zoom = zoom
        self.n_view = n_view
        self.inversion = inversion
        self.reduction = reduction
        self.detach = detach
        self.js_div = JS_Div()
        self.wes_div = Wessertein_Div()

    def forward(self, src_emb, tar_emb, norm=False):
        if norm:
            src_emb = F.normalize(src_emb, dim=1)
            tar_emb = F.normalize(tar_emb, dim=1)

        assert src_emb.shape[0] == tar_emb.shape[0]
        batch_size, device = src_emb.shape[0], src_emb.device
        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.to(device).float()

        p_ab = torch.matmul(src_emb, torch.transpose(src_emb, 0, 1)) / self.tau
        q_ba = torch.matmul(tar_emb, torch.transpose(tar_emb, 0, 1)) / self.tau
        # add self-contrastive
        p_aa = torch.matmul(src_emb, torch.transpose(src_emb, 0, 1)) / self.tau
        q_bb = torch.matmul(tar_emb, torch.transpose(tar_emb, 0, 1)) / self.tau
        p_aa = p_aa - masks * LARGE_NUM
        q_bb = q_bb - masks * LARGE_NUM

        if self.inversion:
            p_ab = torch.cat([p_ab, q_bb], dim=1)
            q_ba = torch.cat([q_ba, p_aa], dim=1)
        else:
            p_ab = torch.cat([p_ab, p_aa], dim=1)
            q_ba = torch.cat([q_ba, q_bb], dim=1)

        # param 1 need to log_softmax, param 2 need to softmax
        loss_a = F.kl_div(F.log_softmax(p_ab, dim=1), F.softmax(q_ba.detach(), dim=1), reduction="none")
        loss_b = self.js_div(p_ab, q_ba.detach())

        if self.reduction == "mean":
            loss_a = loss_a.mean()
            loss_b = loss_b.mean()
        elif self.reduction == "sum":
            loss_a = loss_a.sum()
            loss_b = loss_b.sum()
        # The purpose of the zoom is to narrow the range of losses
        return self.zoom * (loss_a + loss_b) / 2

class ICLLoss(nn.Module):

    def __init__(self, tau=0.05, zoom=0.1, ab_weight=0.5, n_view=2, inversion=False):
        super(ICLLoss, self).__init__()
        self.tau = tau
        self.alpha = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view
        self.zoom = zoom
        self.inversion = inversion
        # self.ialloss = IALLoss()

    def softXEnt(self, logits, target):
        # target/logits : (N, 2 * N)
        logprobs = F.log_softmax(logits, dim=1)
        loss = - (target * logprobs).mean()
        return loss

    def forward(self, emb1, emb2, norm=False):
        # emb1 : (N, dim), emb2 : (N, dim)

        # Get (normalized) hidden1 and hidden2.
        if norm:
            hidden1 = F.normalize(emb1, dim=1)
            hidden2 = F.normalize(emb2, dim=1)
        else:
            hidden1 = emb1
            hidden2 = emb2

        LARGE_NUM = 1e9
        batch_size, device = hidden1.shape[0], hidden1.device # N

        hidden1_large = hidden1 # shape is (N, dim)
        hidden2_large = hidden2 # shape is (N, dim)

        num_classes = batch_size * self.n_view # 2 * N
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes) # (N, 2 * N)
        labels = labels.to(device)

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size) # (N, N)
        masks = masks.to(device).float()

        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / self.tau # (N, N)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / self.tau # (N, N)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / self.tau # (N, N)
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / self.tau # (N, N)

        if self.inversion:
            logits_a = torch.cat([logits_ab, logits_bb], dim=-1) # (N, 2 * N)
            logits_b = torch.cat([logits_ba, logits_aa], dim=-1) # (N, 2 * N)
        else:
            logits_a = torch.cat([logits_ab, logits_aa], dim=-1) # (N, 2 * N)
            logits_b = torch.cat([logits_ba, logits_bb], dim=-1) # (N, 2 * N)

        loss_a = self.softXEnt(logits_a, labels)
        loss_b = self.softXEnt(logits_b, labels)

        return self.zoom * (loss_a + loss_b) / 2