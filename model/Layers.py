import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch import einsum
import inspect


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        del ctx, grad_output, a, b, grad_a_dense, edge_idx
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class MultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer
    """

    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, adj):
        output = []
        for i in range(self.n_head):
            N = input.size()[0]
            edge = adj._indices()
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)  # edge: 2*D x E
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze()))  # edge_e: 1 x E

            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda() if next(
                self.parameters()).is_cuda else torch.ones(size=(N, 1)))  # e_rowsum: N x 1
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)  # edge_e: 1 x E

            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)

            output.append(h_prime.unsqueeze(0))

        output = torch.cat(output, dim=0)
        
        del input, adj, edge, h, edge_h, edge_e, e_rowsum, h_prime
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(
                self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(
                self.n_head) + ' heads'

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)  # spmm does sparse matrix multiplication

        del input, adj, support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.l2(x)
        return x

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        del adj

        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(x)
        del adj
        return x


class ReLUSquared(nn.Module):
    "https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py#L121"
    def forward(self, x):
        return F.relu(x) ** 2

class LaplacianAttnFn(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """
    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

class OffsetScale(nn.Module):
    "https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py#L121"
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        # x : (N, dim)
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta # (N, 2, dim)
        return out.unbind(dim=-2)

class GAU(nn.Module):
    def __init__(self, dim, query_key_dim=200, expansion_factor=1, add_residual=True, dropout=0., laplace_attn_fn=False):
        super().__init__()
        self.dim = dim
        hidden_dim = int(expansion_factor * dim)

        self.dropout = nn.Dropout(dropout)

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads = 2)

        self.to_out = nn.Linear(hidden_dim, dim)

        self.add_residual = add_residual

    def forward(self, x, mask=None):
        # x : (N, dim)

        v, gate = self.to_hidden(x).chunk(2, dim=-1) # v : (N, hidden_dim), gate : (N, hidden_dim)

        qk = self.to_qk(x) # (N, query_key_dim)
        q, k = self.offsetscale(qk) # (N, query_key_dim)， (N, query_key_dim)

        sim = einsum('i d, j d -> i j', q, k) /  math.sqrt(self.dim) # (N, N)

        attn = self.attn_fn(sim) # (N, N)
        attn = self.dropout(attn) # (N, N)

        if mask is not None:
            attn = attn * mask # (N, N)

        out = einsum('i j, j d -> i d', attn, v) # (N, hidden_dim)
        out = out * gate # (N, hidden_dim)
        out = self.to_out(out) # (N, dim)

        del v, gate, qk, q, k, sim, attn

        if self.add_residual:
            out = out * x # (N, dim)

        return out

class GraphAttention(nn.Module):
    "This is a class of like graph attention"
    def forward(self, node, adj):
        # node : (N, M); adj : (N, N)
        z_att = torch.matmul(adj, node) # (N, M)
        z_att = F.relu(z_att)

        node = node + z_att

        del z_att, adj

        return node

class MultiModalFusion(nn.Module):
    "https://github.com/lzxlin/MCLEA/blob/main/src/models.py"
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        del embs, weight_norm
        return joint_emb

class Multi_Modal_Aggregator(nn.Module):
    def __init__(self, in_features, out_features, modal_num=3, bias=False):
        super(Multi_Modal_Aggregator, self).__init__()
        self.in_features = in_features
        self.gau = GAU(modal_num * in_features)
        self.weight_Q  = nn.Linear(in_features, in_features) # (2*M, 1)
        self.weight_K  = nn.Linear(modal_num * in_features, in_features) # (2*M, 1)
        self.weight_V  = nn.Linear(modal_num * in_features, in_features) # (2*M, 1)
        self.graphattention = GraphAttention()
        self.weight = nn.Linear(in_features, out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, node, support):
        Q = self.weight_Q(node) # (N, 2 * dim)
        support = self.gau(support, F.softmax(torch.matmul(support, support.T), dim=-1))
        K = self.weight_K(support)
        V = self.weight_V(support)

        B1 = K * self.graphattention(Q, F.softmax(torch.matmul(K, K.T), dim=-1)) * Q
        B2 = V * self.graphattention(Q, F.softmax(torch.matmul(V, V.T), dim=-1)) * Q
        gt = F.sigmoid(self.weight(B1)) # (N, 2 * dim)
        Gs = F.softmax(torch.matmul((gt * K), ((1 - gt) * Q).T) / math.sqrt(self.in_features), dim=-1) # (N, N)
        output = F.sigmoid(B1 * B1 * torch.matmul(Gs, B2))
        output = F.normalize(output, p=2, dim=-1)

        del Q, support, K, V, B1, B2, gt, Gs
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class TransE(nn.Module):
    def __init__(self, margin, norm=1, reduction='mean') -> None:
        super(TransE, self).__init__()
        self.margin = margin
        self.norm = norm
        self.reduction = reduction  

    def _distance(self, heads, relations, tails):
        # cos_sim = torch.cosine_similarity(heads + relations, tails, dim=1) # (N)
        score = (heads + relations - tails).norm(p=self.norm, dim=1) # (N)
        # score = (2 * score - torch.mul(cos_sim, score)) # (N)
        return score
    
    def forward(self, r_p_h, r_p_t, r_p_r, r_n_h, r_n_t, r_n_r):
        # r_p_h, r_p_t, r_p_r, r_n_h, r_n_t, r_n_r : (N, dim)
        pos_score_r = self._distance(r_p_h, r_p_r, r_p_t) # h + r -t = 0 Transe
        neg_score_r = self._distance(r_n_h, r_n_r, r_n_t) # h + r -t = 0
        neg_score_r = torch.mean(neg_score_r.reshape(pos_score_r.shape[0], -1), dim=-1)
        
        if self.reduction == 'sum':
            relation_loss = torch.sum(F.relu(self.margin + pos_score_r - neg_score_r))
        else:
            relation_loss = torch.mean(F.relu(self.margin + pos_score_r - neg_score_r))

        return relation_loss

class RGCN_Atten(nn.Module):
    "This is inspired by light gcn and rgan"
    def __init__(self, input_dim, hidden_dim):
        super(RGCN_Atten, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_dim, hidden_dim))
        nn.init.xavier_normal_(self.weight.data)

    def forward(self, edge, mask):
        dim = edge.shape[-1]
        edge = torch.mul(edge, mask) # (N, M, dim)
        edge_h = torch.matmul(edge, self.weight) # (N, M, dim)
        att = torch.matmul(edge, edge.transpose(1, 2)) / math.sqrt(dim) # (N, M, M)
        # att = F.softmax(att, dim=-1)
        edge = torch.matmul(att, edge_h) # (N, M, dim)
        output = torch.mean(edge, dim=1) # (N, dim)
        del edge, mask, edge_h, att
        return output

class Classifier(nn.Module):
    def __init__(self, n_input, hidden_1, hidden_2, dropout=0.5, n_classes=1):
        super(Classifier, self).__init__()
        self.dropout = dropout
        self.n_classes = n_classes

        self.fc1 = nn.Linear(n_input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, n_classes)

    # def forward(self, adj):
    #     adj = self.fc1(adj)
    #     # adj = F.normalize(adj, 2, -1)
    #     frame = inspect.currentframe()
    #     adj = F.relu(self.fc2(adj), inplace=True)
    #     adj = self.fc3(adj)
    #     adj = F.log_softmax(adj, dim=2).view(-1, self.n_classes)
    #     del frame
    #     return adj

    def forward(self, adj):
        adj = F.relu(self.fc1(adj), inplace=True)
        adj = F.dropout(adj, self.dropout, training=self.training)
        frame = inspect.currentframe()
        adj = F.relu(self.fc2(adj), inplace=True)
        adj = F.dropout(adj, self.dropout, training=self.training)
        adj = self.fc3(adj)
        adj = F.log_softmax(adj, dim=2).view(-1, self.n_classes)
        del frame
        return adj
