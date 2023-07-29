
import torch
import numpy as np
import random
import math
from torch.utils import data
import multiprocessing as mp

class Dataset(data.Dataset):
    def __init__(self, name_list, label_data_path):
        self.name_list = name_list
        self.label_data_path = label_data_path

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        case_name = self.name_list[index]
        edge_label = torch.load('{}/{}'.format(self.label_data_path, case_name + ".class")) # [[], [], ...] shape is N * N.
        N = edge_label.shape[0]

        link_label = torch.load('{}/{}'.format(self.label_data_path, case_name + ".link")) # [[], [], ...] shape is N1 * M.
        M = link_label.shape[1]

        triples = torch.load('{}/{}'.format(self.label_data_path, case_name + ".triples")) # [name1, name2, ...] shape is 1 * N.

        adj = torch.load('{}/{}'.format(self.label_data_path, case_name + ".adj")) # [name1, name2, ...] shape is 1 * N.

        entity_name_ids = torch.load('{}/{}'.format(self.label_data_path, case_name + ".id")) # [name_id1, name_id2, ...] shape is 1 * N.
        return N, case_name, edge_label, M, link_label, entity_name_ids, triples, adj

def process_loaded_data(data, device, entity_list=None, batch_size=1, batch_threads_num=0, neg_triples_num=1, is_fixed_size=False, neighbor=None, max_try=10):
    """
    process the data when input
    Input:
    data : [N, edge_label, case_name], max_size : int
    Output:
    N : tensor, edge_label : tensor, shape is (1, N*N), class_weight : tensor, case_name[0]
    """
    N, case_name, edge_label, M, link_label, entity_name_ids, triples, adj = data

    adj = adj.squeeze(0)
    adj = torch.sparse_coo_tensor(indices=torch.nonzero(adj).t(),
                                            values=adj[adj != 0],
                                            size=adj.size())

    edge_label = edge_label.view(1, N * N)

    link_label = link_label.view(-1, M)
    ent_link, pos_neg_link = link_label[:,0], link_label[:,1:]

    label = torch.zeros_like(pos_neg_link)
    label[:,0] = 1.0
    label = label.to(torch.float)

    # class_weight = torch.Tensor([1, float(N * N - edge_label.sum()) / float(edge_label.sum())])
    class_weight = torch.Tensor([1, 1 + 0.5/(1 + math.exp(-(float(N * N - edge_label.sum()) / float(edge_label.sum()))))])

    entity_name_ids = torch.tensor(entity_name_ids).long().view(-1)
    
    if entity_list:
        entity_list = set(entity_list)
    else:
        entity_list = set(entity_name_ids.numpy().tolist())

    ''
    manager = mp.Manager()
    relation_batch_queue = manager.Queue()
    triples = {tuple(triple) for triple in triples[0].numpy().tolist()}
    relation_triple_steps = int(math.ceil(len(triples) / batch_size))
    relation_step_tasks = task_divide(list(range(relation_triple_steps)), batch_threads_num)
    for steps_task in relation_step_tasks:
        mp.Process(target=generate_relation_triple_batch_queue,
                    args=(triples, entity_list, batch_size, steps_task, relation_batch_queue, neg_triples_num, is_fixed_size, neighbor, max_try)).start()
    pos_neg_triples_list = []
    for _ in range(relation_triple_steps):
        batch_pos, batch_neg = relation_batch_queue.get()
        rel_p_h = torch.LongTensor([x[0] for x in batch_pos]).to(device)
        rel_p_r = torch.LongTensor([x[1] for x in batch_pos]).to(device)
        rel_p_t = torch.LongTensor([x[2] for x in batch_pos]).to(device)
        rel_n_h = torch.LongTensor([x[0] for x in batch_neg]).to(device)
        rel_n_r = torch.LongTensor([x[1] for x in batch_neg]).to(device)
        rel_n_t = torch.LongTensor([x[2] for x in batch_neg]).to(device)
        pos_neg_triples_list.append([rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t])
    return N, M, edge_label.to(device), ent_link.to(device), pos_neg_link.to(device), label.to(device), class_weight.to(device), entity_name_ids.to(device), pos_neg_triples_list, adj.to(device), case_name[0]

def task_divide(idx, n):
    """
    devide tasks, each task consists of batches
    Input:
    idx : list(range(relation_triple_steps)), n : number
    Output:
    [[batch1, batch2, ...], ...]
    """
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks

def generate_relation_triple_batch_queue(triple_set, entity_list, batch_size, steps, out_queue, neg_triples_num, is_fixed_size=False, neighbor=None, max_try=10):
    """
    generate the triples' queue based on relations
    Input:
    triple_set : {(h_id, r_id, t_id), ...}
    entity_list : [id, ...]
    batch_size : number
    out_queue : type is queue
    neg_triples_num : number
    Output:
    None
    """
    for step in steps: # steps : [batch1, batch2, ...]
        # print(step)
        pos_batch, neg_batch = generate_relation_triple_batch(triple_set, entity_list, batch_size, step, neg_triples_num, is_fixed_size=is_fixed_size, \
        neighbor=neighbor, max_try=max_try)
        out_queue.put((pos_batch, neg_batch))
    exit(0)

def generate_relation_triple_batch(triple_set, entity_list, batch_size, step, neg_triples_num, is_fixed_size=False, neighbor=None, max_try=10):
    """
    Input:
    triple_set : {(h_id, r_id, t_id), ...}
    entity_list : [id, ...]
    batch_size : number
    out_queue : type is queue
    neg_triples_num : number
    Output:
    pos_batch : [(h_id, r_id, t_id), ...]
    neg_batch : [(h_id, r_id, t_id), ...]
    """
    pos_batch = generate_pos_triples(triple_set, batch_size, step, is_fixed_size=is_fixed_size)
    neg_batch = generate_neg_triples_fast(pos_batch, triple_set, entity_list, neg_triples_num, neighbor=neighbor, max_try=max_try)
    return pos_batch, neg_batch

def generate_pos_triples(triples, batch_size, step, is_fixed_size=False):
    """
    generate positive triples
    Input:
    triples : [(h_id, r_id, t_id), ...]
    batch_size : number 
    step : number and is a batch index
    Output:
    pos_batch : [(h_id, r_id, t_id), ...]
    """
    if type(triples) != list:
        triples = list(triples)
    start = step * batch_size
    end = start + batch_size
    if end > len(triples):
        end = len(triples)
    pos_batch = triples[start: end]
    if is_fixed_size and len(pos_batch) < batch_size:
        pos_batch += triples[len(pos_batch) - batch_size : ]
        assert len(pos_batch) == batch_size
    return pos_batch

def generate_neg_triples_fast(pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
    """
    generate negative triples
    Input:
    pos_batch : [(h_id, r_id, t_id), ...]
    all_triples_set : {(h_id, r_id, t_id), ...}
    entities_list : [id, ...] 
    neg_triples_num : number
    Output:
    neg_batch : [(h_id, r_id, t_id), ...]
    """
    if type(all_triples_set) != set:
        all_triples_set = set(all_triples_set)
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, relation, tail in pos_batch:
        neg_triples = list()
        nums_to_sample = neg_triples_num
        head_candidates = entities_list - neighbor.get(head, set([]))
        tail_candidates = entities_list - neighbor.get(tail, set([]))
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg_triples)
                break
            else:
                i_neg_triples = list(i_neg_triples - all_triples_set)
                neg_triples += i_neg_triples
            if len(neg_triples) == neg_triples_num:
                break
            else:
                nums_to_sample = neg_triples_num - len(neg_triples)
        assert len(neg_triples) == neg_triples_num
        neg_batch.extend(neg_triples)
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch

def early_stop(flag1, flag2, flag):
    """
    # early stop

    flag1, flag2, flag

    """
    if flag <= flag2 <= flag1:
        return flag2, flag, True
    else:
        return flag2, flag, False