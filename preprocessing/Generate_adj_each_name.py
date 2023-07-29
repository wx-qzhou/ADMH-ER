from os.path import join
from utils.data_utils import load_json
import os
import numpy as np
import scipy.sparse as sp
import torch
from utils.Path_file import *

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_adjr1(ent_size, triples, dis_entities_index_dict, norm=False):
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if tri[0] in dis_entities_index_dict and tri[2] in dis_entities_index_dict:
            if (tri[0], tri[2]) not in M:
                M[(tri[0], tri[2])] = 0
            M[(tri[0], tri[2])] += 1
    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))  # 关系逆
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)
    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj)).to_dense()
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size])).to_dense()
        return M

def get_adjr2(ent_size, triples, dis_entities_index_dict, norm=False):
    ent_neighbor = {}
    for h, r, t in triples:
        if h in dis_entities_index_dict:
            neighbors = ent_neighbor.get(dis_entities_index_dict[h], set())
            neighbors.add(str(t) + "_")
            ent_neighbor[dis_entities_index_dict[h]] = neighbors
        if t in dis_entities_index_dict:
            neighbors = ent_neighbor.get(dis_entities_index_dict[t], set())
            neighbors.add(str(h) + "_")
            ent_neighbor[dis_entities_index_dict[t]] = neighbors
    ent_neighbor = list(ent_neighbor.items())
    M = {}
    for i in range(len(ent_neighbor) - 1):
        for j in range(i + 1, len(ent_neighbor)):  
            if len(ent_neighbor[i][1] & ent_neighbor[j][1]) != 0:
                M[(ent_neighbor[i][0], ent_neighbor[j][0])] = len(ent_neighbor[i][1] & ent_neighbor[j][1])
    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))  # 关系逆
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)
    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj)).to_dense()
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size])).to_dense()
        return M

def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f

def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if

def get_adjr(ent_size, triples, dis_entities_index_dict, norm=False):
    r2f = func(triples)
    r2if = ifunc(triples)

    ent_neighbor_h = {}
    ent_neighbor_t = {}
    for h, r, t in triples:
        if h in dis_entities_index_dict:
            neighbors = ent_neighbor_h.get(dis_entities_index_dict[h], set())
            neighbors.add((r, t))
            ent_neighbor_h[dis_entities_index_dict[h]] = neighbors
        if t in dis_entities_index_dict:
            neighbors = ent_neighbor_t.get(dis_entities_index_dict[t], set())
            neighbors.add((r, h))
            ent_neighbor_t[dis_entities_index_dict[t]] = neighbors
    ent_neighbor_h = list(ent_neighbor_h.items())
    ent_neighbor_t = list(ent_neighbor_t.items())
    M = {}
    for i in range(len(ent_neighbor_h) - 1):
        for j in range(i + 1, len(ent_neighbor_h)):  
            if len(ent_neighbor_h[i][1] & ent_neighbor_h[j][1]) != 0:
                join_rel_ent = ent_neighbor_h[i][1] & ent_neighbor_h[j][1]
                weight = 0
                for rel, ent in join_rel_ent:
                    weight += max(r2if[rel], 0.3)
                M[(ent_neighbor_h[i][0], ent_neighbor_h[j][0])] = weight
    for i in range(len(ent_neighbor_t) - 1):
        for j in range(i + 1, len(ent_neighbor_t)):  
            if len(ent_neighbor_t[i][1] & ent_neighbor_t[j][1]) != 0:
                join_rel_ent = ent_neighbor_t[i][1] & ent_neighbor_t[j][1]
                weight = 0
                for rel, ent in join_rel_ent:
                    weight += max(r2f[rel], 0.3)
                M[(ent_neighbor_t[i][0], ent_neighbor_t[j][0])] = weight

    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)
    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj)).to_dense()
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size])).to_dense()
        return M

def read_mapping_id(file_path):
    ent_ids = load_json(join(file_path, "ent_ids_dict.json"))
    rel_ids = load_json(join(file_path, "rel_ids_dict.json"))
    return ent_ids, rel_ids

def main_child_graph_adj(datasets_file, assessments_file):
    ent_ids, rel_ids = read_mapping_id(datasets_file)
    dis_name = set()
    for file_name in os.listdir(assessments_file):
        dis_name.add(file_name.split(".")[0])
    for file_name in dis_name:
        print('getting a sparse tensor r_adj of name {}'.format(file_name))
        raw_child_graphs = torch.load('{}/{}'.format(assessments_file, file_name + ".triples")).tolist() # [name1, name2, ...] shape is 1 * N.
        dis_entities_list = sorted([ent_ids[ent] for ent in list(load_json(join(assessments_file, file_name + ".json")))])

        dis_entities_index_dict = dict([(dis_entities_list[i], i) for i in range(len(dis_entities_list))])

        M = get_adjr(len(dis_entities_list), raw_child_graphs, dis_entities_index_dict, norm=False)
        torch.save(
            M,
            '{}/{}'.format(assessments_file, file_name + ".adj")
        )

if __name__ == "__main__":
    args = read_args("config.json")
    main_child_graph_adj(args.Preprocessing_data_file, args.assessments_file)