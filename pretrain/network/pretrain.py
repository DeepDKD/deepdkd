# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pretrain.network.head import *
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

class Pretrain(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.996, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.996)
        T: softmax temperature (default: 0.07)
        """
        super(Pretrain, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.head = ProjectionHead(dim_in=2048, dim_out=dim, dim_hidden=4096)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def build_connected_component(self, dist):
        b = dist.size(0)
        dist = dist - torch.eye(b, b, device='cuda') * 2
        x = torch.arange(b, device='cuda').unsqueeze(1).repeat(1,1).flatten()
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])
        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels, device='cuda')
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is not None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * diagnal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, im_q, im_k, t=0.07):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # compute query features
        f_q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(f_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            f_k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(f_k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        b = im_q.size(0)
        first_half_label = torch.arange(b-1, 2*b-1).long().cuda()
        second_half_label = torch.arange(0, b).long().cuda()
        labels = torch.cat([first_half_label, second_half_label])

        feat_q = F.normalize(self.head(f_q))
        feat_k = F.normalize(self.head(f_k))
        all_feat_q = concat_all_gather(feat_q)
        all_feat_k = concat_all_gather(feat_k)
        all_bs = all_feat_q.size(0)

        mask1_list = []
        mask2_list = []
        if rank == 0:
            mask1 = self.build_connected_component(all_feat_q @ all_feat_q.T).float()
            mask2 = self.build_connected_component(all_feat_k @ all_feat_k.T).float()
            mask1_list = list(torch.chunk(mask1, world_size))
            mask2_list = list(torch.chunk(mask2, world_size))
            mask1 = mask1_list[0]
            mask2 = mask2_list[0]
        else:
            mask1 = torch.zeros(b, all_bs, device='cuda')
            mask2 = torch.zeros(b, all_bs, device='cuda')
        torch.distributed.scatter(mask1, mask1_list, 0)
        torch.distributed.scatter(mask2, mask2_list, 0)

        diagnal_mask = torch.eye(all_bs, all_bs, device='cuda')
        diagnal_mask = torch.chunk(diagnal_mask, world_size)[rank]
        graph_loss =  self.sup_contra(feat_q @ all_feat_q.T / t, mask2, diagnal_mask)
        graph_loss += self.sup_contra(feat_k @ all_feat_k.T / t, mask1, diagnal_mask)
        graph_loss *= 0.2
        return logits, labels, graph_loss



# utils
@torch.no_grad()
def concat_all_gather(tensor, replace=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    rank = torch.distributed.get_rank()
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    if replace:
        tensors_gather[rank] = tensor
    other = torch.cat(tensors_gather, dim=0)
    return other


