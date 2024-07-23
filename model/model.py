import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .Layers import Multi_Modal_Aggregator, TransE, RGCN_Atten, MultiModalFusion, Classifier, GAT, GCN
from .Loss import InfoNCE

class MSDIS(nn.Module):
    """
    The main class 
    """
    def __init__(self, kg, args):
        super(MSDIS, self).__init__()
        self.dis_entities_len = kg.dis_entities_len
        self.ent_num = kg.ent_num
        self.rel_num = kg.rel_num
        self.modality_num = 5
        self.kg = kg
        self.args = args
        self.n_layers = 2

        'relation'
        self.ent_embed_ = nn.Embedding(self.dis_entities_len, self.args.dim)
        self.ent_embed = nn.Embedding(self.ent_num, self.args.hidden_dim)
        self.rel_embed = nn.Embedding(self.rel_num, self.args.hidden_dim)
        self.rel_index, self.ent_index, self.emask, self.rmask = kg.entid_rel_list, kg.entid_ent_list, kg.e_mask, kg.r_mask
        nn.init.xavier_normal_(self.ent_embed.weight.data)
        nn.init.xavier_normal_(self.rel_embed.weight.data)
        # self.transE_forward = TransE(self.args.margin)

        'Gat'
        self.hidden_units = self.args.hidden_units
        self.heads = self.args.heads
        self.dropout = self.args.dropout
        self.attn_dropout = self.args.attn_dropout
        self.instance_normalization = self.args.instance_normalization
        self.structure_encoder = self.args.structure_encoder

        self.n_units = [int(x) for x in self.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.heads.strip().split(",")]
        
        # structure encoder
        if self.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.dropout)
        elif self.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=self.dropout,
                                         attn_dropout=self.attn_dropout,
                                         instance_normalization=self.instance_normalization, diag=True)

        'RGCN'
        self.ones_zeros = nn.Embedding(2, self.args.hidden_dim)
        self.ones_zeros.weight.data[0] = torch.zeros((1, self.args.hidden_dim))
        self.ones_zeros.weight.data[1] = torch.ones((1, self.args.hidden_dim))
        self.ones_zeros.requires_grad = False

        self.ent_atten_blocks = nn.ModuleList([RGCN_Atten(self.args.hidden_dim, self.args.hidden_dim) for _ in range(self.n_layers)])
        self.rel_atten_blocks = nn.ModuleList([RGCN_Atten(self.args.hidden_dim, self.args.hidden_dim) for _ in range(self.n_layers)])
        self.sum_weight = nn.Parameter(torch.FloatTensor([1] * self.n_layers),)

        'image'
        self.img_embed = nn.Embedding.from_pretrained(torch.FloatTensor(kg.images_list))
        # self.img_embed.requires_grad = False

        'attr'
        self.attr_embed = nn.Embedding.from_pretrained(torch.FloatTensor(kg.attr_list))
        # self.attr_embed.requires_grad = False

        'text'
        self.txt_embed = nn.Embedding.from_pretrained(torch.FloatTensor(kg.txt_list))
        # self.txt_embed.requires_grad = False

        'FC layer'
        self.fc_e_ = nn.Linear(self.args.dim, self.args.dim)
        self.fc_e = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.fc_r = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.fc_i = nn.Linear(self.args.i_dim, self.args.dim)
        self.fc_a = nn.Linear(self.args.a_dim, self.args.dim)
        self.fc_t = nn.Linear(self.args.t_dim, self.args.dim)
        nn.init.xavier_normal_(self.fc_e_.weight.data)
        nn.init.xavier_normal_(self.fc_e.weight.data)
        nn.init.xavier_normal_(self.fc_r.weight.data)
        nn.init.xavier_normal_(self.fc_i.weight.data)
        nn.init.xavier_normal_(self.fc_a.weight.data)
        nn.init.xavier_normal_(self.fc_t.weight.data)

        'multimodal fusion'
        self.MMA = Multi_Modal_Aggregator(self.args.dim, self.args.hidden_dim, self.args.dim, self.modality_num-2)
        self.modal_fusion = MultiModalFusion(self.modality_num - 1)
        self.modal_fusion2 = MultiModalFusion(3)
        self.info_nce = InfoNCE(temperature=self.args.tau, reduction=self.args.reduction)

        'classification'
        self.classifier = Classifier(2 * (self.modality_num * self.args.dim + 2 * self.args.hidden_dim) + self.modality_num + 1, 2 * self.args.dim, self.args.dim, dropout=self.args.class_dropout, n_classes=2)

    def e_rep(self, e):
        return F.normalize(self.fc_e(self.ent_embed(e)), 2, -1)
    
    def e_rep_(self, e):
        return F.normalize(self.fc_e_(self.ent_embed_(e)), 2, -1)

    def r_rep(self, e):
        return F.normalize(self.fc_r(self.rel_embed(e)), 2, -1)

    def i_rep(self, e):
        return F.normalize(self.fc_i(self.img_embed(e)), 2, -1)
    
    def a_rep(self, e):
        return F.normalize(self.fc_a(self.attr_embed(e)), 2, -1)
    
    def t_rep(self, e):
        return F.normalize(self.fc_t(self.txt_embed(e)), 2, -1)
    
    def RGCN_Aggregation(self, e):
        # e : (N, )
        device = e.device
        edge_r, edge_e, emask, rmask = self.r_rep(self.rel_index[e].to(device)), self.e_rep(self.ent_index[e].to(device)), \
        self.ones_zeros(self.emask[e].to(device)), self.ones_zeros(self.rmask[e].to(device)) # (N, max_len, dim)

        edge_e_list = []
        for ent_atten in self.ent_atten_blocks:
            edge_e_list.append(ent_atten.forward(edge_e, emask))

        edge_r_list = []
        for rel_atten in self.rel_atten_blocks:
            edge_r_list.append(rel_atten.forward(edge_r, rmask)) # (N, M)

        edge_e = torch.mean(torch.stack([self.sum_weight[idx] * edge_e_list[idx] for idx in range(len(edge_e_list))], dim=1), dim=1)
        edge_r = torch.mean(torch.stack([self.sum_weight[idx] * edge_r_list[idx] for idx in range(len(edge_r_list))], dim=1), dim=1)

        edge_er = torch.cat((edge_e, edge_r), dim=-1)  # (N, 2 * dim)
        del edge_r, edge_e, emask, rmask, edge_r_list, edge_e_list, e
        return edge_er

    def similarity_score(self, embed1, embed2, metric='cosine'):
        if metric == 'cosine':
            score = torch.cosine_similarity(embed1, embed2, dim=-1)
        if metric == 'euclidean':
            score = 1 - F.pairwise_distance(embed1, embed2, p=2)
        if metric == 'manhattan':
            score = 1 - F.normalize(torch.cdist(embed1, embed2, p=1), 2, -1)
        if metric == "inner":
            if len(embed1.shape) == 3:
                embed1 = torch.mean(embed1, dim=1).unsqueeze(-1)
                score = F.softmax(torch.bmm(embed2, embed1).squeeze(-1), dim=-1)
            else:
                score = F.softmax(torch.mm(embed1, embed2.T).squeeze(-1), dim=-1) 
        del embed1, embed2
        return score

    def embeddings_based_index_(self, hidden_e, e, p, n, N, M):
        e_r_embed = torch.index_select(hidden_e, 0, e)   
        p_r_embed = torch.index_select(hidden_e, 0, p)
        n_r_embed = torch.index_select(hidden_e, 0, n)

        e_r_embed = e_r_embed.reshape(N, -1)
        p_r_embed = p_r_embed.reshape(N, -1)
        n_r_embed = n_r_embed.reshape(N, M-1, -1)
        del hidden_e, e, p, n, N, M
        return e_r_embed, p_r_embed, n_r_embed

    def contrastive_Learning(self, e, pn, all_e, sparse_adj):
        # e : (N,), pn : (N, M)
        N, M = e.shape[0], pn.shape[1]

        hidden_e, hidden_r, hidden_i, hidden_a, hidden_t = self.obtain_embeddings(all_e, sparse_adj) # (N1, dim) 
        hidden_all = self.multimodal(hidden_e, hidden_r, hidden_i, hidden_t, hidden_a)

        p, n = pn[:,0], pn[:,1:].reshape(-1)

        e_e_embed, p_e_embed, n_e_embed = self.embeddings_based_index_(hidden_e, e, p, n, N, M) # (N * M, dim) 
        e_r_embed, p_r_embed, n_r_embed = self.embeddings_based_index_(hidden_r, e, p, n, N, M) # (N * M, dim) 
        e_i_embed, p_i_embed, n_i_embed = self.embeddings_based_index_(hidden_i, e, p, n, N, M) # (N * M, dim) 
        e_t_embed, p_t_embed, n_t_embed = self.embeddings_based_index_(hidden_t, e, p, n, N, M) # (N * M, dim)
        e_a_embed, p_a_embed, n_a_embed = self.embeddings_based_index_(hidden_a, e, p, n, N, M) # (N * M, dim)
        e_all_embed, p_all_embed, n_all_embed = self.embeddings_based_index_(hidden_all, e, p, n, N, M) # (N * M, dim)

        loss_e = self.info_nce(e_e_embed, p_e_embed, n_e_embed)
        loss_r = self.info_nce(e_r_embed, p_r_embed, n_r_embed)
        loss_i = self.info_nce(e_i_embed, p_i_embed, n_i_embed)
        loss_t = self.info_nce(e_t_embed, p_t_embed, n_t_embed)
        loss_a = self.info_nce(e_a_embed, p_a_embed, n_a_embed)
        loss_all = self.info_nce(e_all_embed, p_all_embed, n_all_embed)

        
        del e, pn, all_e, sparse_adj, p, n
        del hidden_e, hidden_r, hidden_i, hidden_a, hidden_t, hidden_all
        del e_e_embed, e_r_embed, e_i_embed, e_t_embed, e_a_embed, e_all_embed
        del p_e_embed, p_r_embed, p_i_embed, p_t_embed, p_a_embed, p_all_embed
        del n_e_embed, n_r_embed, n_i_embed, n_t_embed, n_a_embed, n_all_embed
        gc.collect()
        return [loss_e, loss_r, loss_i, loss_t, loss_a], loss_all * self.args.Lambda

    def multimodal(self, e_e_embed, e_r_embed, e_i_embed, e_t_embed, e_a_embed):
        
        hidden_all = self.modal_fusion([e_r_embed, e_i_embed, e_t_embed, e_a_embed])
        output = self.MMA(e_e_embed, hidden_all) # (N1, dim)
        output = self.modal_fusion2([e_e_embed, hidden_all, output]) # (N1, 4 * dim)

        del e_e_embed, e_r_embed, e_i_embed, e_t_embed, e_a_embed, hidden_all
        return output

    def obtain_embeddings(self, e, sparse_adj):
        hidden_e = self.cross_graph_model(self.e_rep_(e), sparse_adj)
        hidden_r = self.RGCN_Aggregation(e) # (N, dim)
        hidden_i = self.i_rep(e) # (N, dim)
        hidden_a = self.a_rep(e) # (N, dim)
        hidden_t = self.t_rep(e) # (N, dim)
        del e, sparse_adj
        return hidden_e, hidden_r, hidden_i, hidden_a, hidden_t
    
    def embeddings_based_index(self, hidden_e, e, pn, N, M):
        e_r_embed = torch.index_select(hidden_e, 0, e)   
        pn_r_embed = torch.index_select(hidden_e, 0, pn)

        e_r_embed = e_r_embed.reshape(N, M, -1)
        pn_r_embed = pn_r_embed.reshape(N, M, -1)
        del hidden_e, e, pn, N, M
        return e_r_embed, pn_r_embed

    # 'TransE'
    # def forward_transe(self, r_p_h, r_p_r, r_p_t, r_n_h, r_n_r, r_n_t):
    #     r_p_h = self.e_rep(r_p_h)
    #     r_p_t = self.e_rep(r_p_t)
    #     r_p_r = self.r_rep(r_p_r)
    #     r_n_h = self.e_rep(r_n_h)
    #     r_n_t = self.e_rep(r_n_t)
    #     r_n_r = self.r_rep(r_n_r)
    #     relation_loss = self.transE_forward(r_p_h, r_p_t, r_p_r, r_n_h, r_n_t, r_n_r)
    #     del r_p_h, r_p_t, r_p_r, r_n_h, r_n_t, r_n_r
    #     return relation_loss

    'Link'
    def forward_link(self, e, pn, all_e, sparse_adj):
        # e : (N,), pn : (N, M)
        N, M = e.shape[0], pn.shape[1]
        
        hidden_e, hidden_r, hidden_i, hidden_a, hidden_t = self.obtain_embeddings(all_e, sparse_adj) # (N1, dim)

        hidden_all = self.multimodal(hidden_e, hidden_r, hidden_i, hidden_t, hidden_a)
        # reset the shape of index
        e = e.unsqueeze(1).expand_as(pn).reshape(-1) # (N * M)  
        pn = pn.reshape(-1) # (N * M)  

        e_e_embed, pn_e_embed = self.embeddings_based_index(hidden_e, e, pn, N, M) # (N * M, dim) 
        e_r_embed, pn_r_embed = self.embeddings_based_index(hidden_r, e, pn, N, M) # (N * M, dim) 
        e_i_embed, pn_i_embed = self.embeddings_based_index(hidden_i, e, pn, N, M) # (N * M, dim) 
        e_t_embed, pn_t_embed = self.embeddings_based_index(hidden_t, e, pn, N, M) # (N * M, dim)
        e_a_embed, pn_a_embed = self.embeddings_based_index(hidden_a, e, pn, N, M) # (N * M, dim)
        e_all_embed, pn_all_embed = self.embeddings_based_index(hidden_all, e, pn, N, M) # (N * M, dim)

        e_score = self.similarity_score(e_e_embed, pn_e_embed) # (N, M)
        r_score = self.similarity_score(e_r_embed, pn_r_embed) # (N, M)
        i_score = self.similarity_score(e_i_embed, pn_i_embed) # (N, M)
        a_score = self.similarity_score(e_a_embed, pn_a_embed) # (N, M)
        t_score = self.similarity_score(e_t_embed, pn_t_embed) # (N, M)
        
        score = self.similarity_score(e_all_embed, pn_all_embed) # (N, M)

        del e, pn, all_e, sparse_adj
        del hidden_e, hidden_r, hidden_i, hidden_a, hidden_t, hidden_all
        del e_e_embed, e_r_embed, e_i_embed, e_t_embed, e_a_embed, e_all_embed
        del pn_e_embed, pn_r_embed, pn_i_embed, pn_t_embed, pn_a_embed, pn_all_embed
        gc.collect()

        return [e_score, r_score, i_score, a_score, t_score, score]

    'Label'
    def forward(self, e, N, sparse_adj):
        # e : (N, )
        e_e_embed, e_r_embed, e_i_embed, e_a_embed, e_t_embed = self.obtain_embeddings(e, sparse_adj) # (N, dim)

        hidden_all = self.multimodal(e_e_embed, e_r_embed, e_i_embed, e_t_embed, e_a_embed)

        adj = torch.cat([hidden_all.repeat(1, N).view(N * N, -1), hidden_all.repeat(N, 1)], dim=1).view(N, N, -1)
        
        e_score = self.similarity_score(e_e_embed.repeat(1, N).view(N * N, -1), e_e_embed.repeat(N, 1)).reshape(N, N, 1)
        r_score = self.similarity_score(e_r_embed.repeat(1, N).view(N * N, -1), e_r_embed.repeat(N, 1)).reshape(N, N, 1)
        i_score = self.similarity_score(e_i_embed.repeat(1, N).view(N * N, -1), e_i_embed.repeat(N, 1)).reshape(N, N, 1)
        a_score = self.similarity_score(e_a_embed.repeat(1, N).view(N * N, -1), e_a_embed.repeat(N, 1)).reshape(N, N, 1)
        t_score = self.similarity_score(e_t_embed.repeat(1, N).view(N * N, -1), e_t_embed.repeat(N, 1)).reshape(N, N, 1)
        score = self.similarity_score(hidden_all, hidden_all, "inner").reshape(N, N, 1)
        # sparse_adj = sparse_adj.to_dense().reshape(N, N, 1)

        weight_norm = F.softmax(self.modal_fusion.weight, dim=0)
        r_score = weight_norm[0] * r_score
        i_score = weight_norm[1] * i_score
        t_score = weight_norm[2] * t_score
        a_score = weight_norm[3] * a_score

        z = torch.cat([adj, e_score, r_score, i_score, t_score, a_score, score], dim=-1)
        output = self.classifier(z)

        del e, N, sparse_adj
        del e_e_embed, e_r_embed, e_i_embed, e_a_embed, e_t_embed
        del e_score, r_score, i_score, t_score, score, a_score
        del hidden_all, adj, z, weight_norm
        gc.collect()

        return output

class Weight_Sum_Loss(nn.Module):
    """
    weighted sum for loss
    """
    def __init__(self, loss_num):
        super(Weight_Sum_Loss, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor([1] * loss_num),)

    def forward(self, loss_list):
        weight_norm = F.softmax(self.weight, dim=0)

        joint_loss = 0 
        for loss in [weight_norm[idx] * loss_list[idx] for idx in range(len(loss_list))]:
            joint_loss += loss
        del loss_list, weight_norm
        return joint_loss