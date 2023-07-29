import gc
import multiprocessing
import time
import numpy as np
import torch
from model.DataLoad import task_divide

def greedy_alignment(sim_mat, top_k, nums_threads):
    """
    Parameters
    ----------
    sim_mat : matrix_like
        An embedding matrix of size n*m, where n1 is the number of embeddings and d is the dimension.
    top_k : list of integers
        Hits@k metrics for evaluating results.
    nums_threads : int
        The number of threads used to search alignment.

    Returns
    -------
    alignment_rest :  list, pairs of aligned entities
    hits1 : float, hits@1 values for alignment results
    mr : float, MR values for alignment results
    mrr : float, MRR values for alignment results
    """
    t = time.time()
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        rests = list()
        search_tasks = task_divide(list(range(num)), nums_threads)
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
    else:
        mr, mrr, hits = calculate_rank(list(range(num)), sim_mat, top_k, num)
    hits = np.array(hits) / num
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    cost = time.time() - t
    del sim_mat
    gc.collect()
    return hits, mr, mrr, cost

def calculate_rank(idx, sim_mat, top_k, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    for i in range(len(idx)):
        rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        rank_index = np.where(rank == 0)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits

def classify_evaluate(preds, labels, pos_weight=1):
    labels = labels.squeeze().reshape(-1, 1).squeeze()
    zes = torch.Tensor(torch.zeros(labels.shape[0])).type(torch.LongTensor)
    ons = torch.Tensor(torch.ones(labels.shape[0])).type(torch.LongTensor)
    tp = int(((preds >= 0.1) & (labels == ons)).sum())
    fp = int(((preds >= 0.1) & (labels == zes)).sum())
    fn = int(((preds < 0.1) & (labels == ons)).sum())
    tn = int(((preds < 0.1) & (labels == zes)).sum())
    tp = int(pos_weight * tp)
    fn = int(pos_weight * fn)
    
    epsilon = 1e-7
    acc = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return acc, recall, (2 * acc * recall) / (acc + recall + 1e-13), torch.tensor([tp, fp, fn, tn])

# if __name__ == "__main__":
#     # preds = torch.Tensor([1, 0, 1, 0])
#     # labels = torch.LongTensor([1, 1, 0, 0])
#     # print(classify_evaluate(preds, labels))

#     # preds = torch.Tensor([[1, 1, 1, 1]])
#     # preds_ = torch.zeros((11, 4))
#     # preds_[10] = preds
#     # sim_mat = torch.matmul(preds, preds_.T)
#     sim_mat =  torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#     num = sim_mat.shape[0]
#     print(sim_mat)
#     mr, mrr, hits = calculate_rank(list(range(num)), sim_mat, [1, 3, 10], num)

#     hits = np.array(hits) / num
#     for i in range(len(hits)):
#         hits[i] = round(hits[i], 3)

#     print(mr, mrr, hits)

#     print(greedy_alignment(sim_mat, [1, 3, 10], 2))
#     pass