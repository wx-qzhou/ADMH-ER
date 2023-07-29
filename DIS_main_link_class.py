from utils.Path_file import *
from model.KG import KG
import time
import torch
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from model.model import *
from model.DataLoad import Dataset, process_loaded_data, early_stop
from torch.utils import data
from utils.evaluation import greedy_alignment, np, classify_evaluate
import random

def test():
    model.eval()
    wdl1.eval()
    wdl2.eval()
    with torch.no_grad():
        hits, mr, mrr, cost = np.array([0.0] * len(args.top_k)), 0, 0, 0
        total_nums = torch.zeros(4, dtype=int)
        prf_nums = torch.zeros(3, dtype=float)
        loss = 0

        for data_index, data in enumerate(testing_generator):
            N, M, edge_label, ent_link, pos_neg_link, label, class_weight, entity_name_ids, pos_neg_triples_list, sparse_adj, case_name = \
                process_loaded_data(data, device, batch_size=args.batch_size, batch_threads_num=args.batch_threads_num, neg_triples_num=args.neg_triples_num)

            'Link'
            score_list = model.forward_link(ent_link, pos_neg_link, entity_name_ids, sparse_adj)
            link_loss = wdl1([link_criterion(score, label) for score in score_list])

            hits_, mr_, mrr_, cost_ = greedy_alignment(score_list[-1].cpu(), args.top_k, args.test_threads_num)
            hits += hits_
            mr += mr_
            mrr += mrr_
            cost += cost_

            'Classify'
            output = model(entity_name_ids, N, sparse_adj)
            classify_loss = F.nll_loss(output, edge_label.squeeze(), class_weight)

            preds = output.max(1)[1].type_as(edge_label.squeeze()).cpu()
            edge_label = edge_label.cpu()
            acc_val, recall_val, f1_val, nums = classify_evaluate(preds, edge_label.squeeze())
            prf_nums += torch.tensor([acc_val, recall_val, f1_val])
            total_nums += nums

            loss_val = wdl2([link_loss, classify_loss])
            loss += float(loss_val)

            del edge_label, ent_link, pos_neg_link, label, class_weight, entity_name_ids

        hits /= (data_index + 1)
        mr /= (data_index + 1)
        mrr /= (data_index + 1)        
        cost /= (data_index + 1)

        tp, fp, fn, tn = total_nums
        precision = tp / (tp + fp + 1e-14)
        recall = tp / (tp + fn + 1e-14)
        f1 = (precision * recall) ** 0.5

        macro_precision = prf_nums[0] / (data_index + 1)
        macro_recall = prf_nums[1] / (data_index + 1)
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-14)

        loss /= (data_index + 1)

        args.logger.info("Next is Testing.")
        args.logger.info("The loss of testing is {:.4f}".format(loss))
        args.logger.info("Linking results: hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                  format(args.top_k, hits, mr, mrr, cost))
        args.logger.info("Classify results: micro_precision: {:.4f}, micro_recall = {:.4f}, micro_f1 = {:.4f}, time = {:.3f} s".format(\
            precision, recall, f1, cost))
        args.logger.info("Classify results: macro_precision: {:.4f}, macro_recall = {:.4f}, macro_f1 = {:.4f}, time = {:.3f} s".format(\
            macro_precision, macro_recall, macro_f1, cost))

    return mrr if args.stop_metric == 'micro_f1' else hits[-1]

def valid():
    model.eval()
    wdl1.eval()
    wdl2.eval()
    with torch.no_grad():
        hits, mr, mrr, cost = np.array([0.0] * len(args.top_k)), 0, 0, 0
        total_nums = torch.zeros(4, dtype=int)
        prf_nums = torch.zeros(3, dtype=float)
        loss = 0

        for data_index, data in enumerate(validing_generator):
            N, M, edge_label, ent_link, pos_neg_link, label, class_weight, entity_name_ids, pos_neg_triples_list, sparse_adj, case_name = \
                process_loaded_data(data, device, batch_size=args.batch_size, batch_threads_num=args.batch_threads_num, neg_triples_num=args.neg_triples_num)
            
            'Link'
            score_list = model.forward_link(ent_link, pos_neg_link, entity_name_ids, sparse_adj)
            link_loss = wdl1([link_criterion(score, label) for score in score_list])

            hits_, mr_, mrr_, cost_ = greedy_alignment(score_list[-1].cpu(), args.top_k, args.test_threads_num)
            hits += hits_
            mr += mr_
            mrr += mrr_
            cost += cost_

            'Classify'
            output = model(entity_name_ids, N, sparse_adj)
            classify_loss = F.nll_loss(output, edge_label.squeeze(), class_weight)

            preds = output.max(1)[1].type_as(edge_label.squeeze()).cpu()
            edge_label = edge_label.cpu()
            acc_val, recall_val, f1_val, nums = classify_evaluate(preds, edge_label.squeeze())
            prf_nums += torch.tensor([acc_val, recall_val, f1_val])
            total_nums += nums

            loss_val = wdl2([link_loss, classify_loss])
            loss += float(loss_val)

            del edge_label, ent_link, pos_neg_link, label, class_weight, entity_name_ids

        hits /= (data_index + 1)
        mr /= (data_index + 1)
        mrr /= (data_index + 1)        
        cost /= (data_index + 1)

        tp, fp, fn, tn = total_nums
        precision = tp / (tp + fp + 1e-14)
        recall = tp / (tp + fn + 1e-14)
        f1 = (precision * recall) ** 0.5

        macro_precision = prf_nums[0] / (data_index + 1)
        macro_recall = prf_nums[1] / (data_index + 1)
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-14)
        
        loss /= (data_index + 1)

        # print("The loss of validing is {:.4f}".format(loss))
        # print("Linking results: hits@{} = {}, mr = {:.4f}, mrr = {:.4f}, time = {:.3f} s ".
        #           format(args.top_k, hits, mr, mrr, cost))
        # print("Classify results: micro_precision: {:.4f}, micro_recall = {:.4f}, micro_f1 = {:.4f}, time = {:.3f} s".format(\
        #     precision, recall, f1, cost))
        # print("Classify results: macro_precision: {:.4f}, macro_recall = {:.4f}, macro_f1 = {:.4f}, time = {:.3f} s".format(\
        #     macro_precision, macro_recall, macro_f1, cost))

        args.logger.info("The loss of validing is {:.4f}".format(loss))
        args.logger.info("Linking results: hits@{} = {}, mr = {:.4f}, mrr = {:.4f}, time = {:.3f} s ".
                  format(args.top_k, hits, mr, mrr, cost))
        args.logger.info("Classify results: micro_precision: {:.4f}, micro_recall = {:.4f}, micro_f1 = {:.4f}, time = {:.3f} s".format(\
            precision, recall, f1, cost))
        args.logger.info("Classify results: macro_precision: {:.4f}, macro_recall = {:.4f}, macro_f1 = {:.4f}, time = {:.3f} s".format(\
            macro_precision, macro_recall, macro_f1, cost))
    return mrr if args.stop_metric == 'micro_f1' else hits[0]

def train():
    t = time.time()
    max_mrr = 0
    flag1, flag2 = -1, -1
    
    for i in range(1, args.max_epoch + 1):
        start = time.time()
        epoch_loss, epoch_rloss, epoch_lloss, epoch_closs = 0, 0, 0, 0
        model.train()
        
        'label'
        for data_index, data in enumerate(training_generator):
            optimizer.zero_grad()

            'TransE'
            relation_loss, link_loss, classify_loss = 0, 0, 0
            N, M, edge_label, ent_link, pos_neg_link, label, class_weight, entity_name_ids, pos_neg_triples_list, sparse_adj, case_name = \
                process_loaded_data(data, device, entity_list=[i for i in range(kg.ent_num)], batch_size=args.batch_size, batch_threads_num=args.batch_threads_num, \
                neg_triples_num=args.neg_triples_num, neighbor=kg.mutil_htop)
            for pos_neg_triples in pos_neg_triples_list:
                rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t = pos_neg_triples
                relation_loss += model.forward_transe(rel_p_h, rel_p_r, rel_p_t, rel_n_h, rel_n_r, rel_n_t)
            relation_loss /= len(pos_neg_triples_list)

            'Link'
            score_list = model.forward_link(ent_link, pos_neg_link, entity_name_ids, sparse_adj)
            # link_loss = F.nll_loss(score.view(-1), label.view(-1))
            link_loss = wdl1([link_criterion(score, label) for score in score_list])

            'Classify'
            output = model(entity_name_ids, N, sparse_adj)
            classify_loss = F.nll_loss(output, edge_label.squeeze(), class_weight)

            'Final loss'
            loss = wdl2([link_loss, classify_loss, relation_loss])

            ''
            epoch_rloss += relation_loss.item()
            epoch_lloss += link_loss.item()  
            epoch_closs += classify_loss.item()

            'loss backward'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)  # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(wdl1.parameters(), 2.0)  # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(wdl2.parameters(), 2.0)  # 梯度裁剪
            optimizer.step()
            del edge_label, ent_link, pos_neg_link, label, class_weight, entity_name_ids, pos_neg_triples_list

        scheduler1.step()   

        epoch_rloss /= (data_index + 1)
        epoch_lloss /= (data_index + 1)
        epoch_closs /= (data_index + 1)
        epoch_loss += epoch_rloss
        epoch_loss += epoch_lloss
        epoch_loss += epoch_closs
        random.shuffle(kg.id_relation_triples)
        end = time.time()
        print()
        # print('[epoch {}] loss: {:.6f}, relation loss: {:.6f}, link loss:{:.6f}, classify loss:{:.6f}, time: {:.4f}s'.format(i, \
        #     epoch_loss, epoch_rloss, epoch_lloss, epoch_closs, end - start))
        args.logger.info('[epoch {}] loss: {:.6f}, relation loss: {:.6f}, link loss:{:.6f}, classify loss:{:.6f}, time: {:.4f}s'.format(i, \
            epoch_loss, epoch_rloss, epoch_lloss, epoch_closs, end - start))
        if i >= args.start_valid and i % args.eval_freq == 0:
            flag = valid()
            if flag > max_mrr:
                torch.save(model.state_dict(), out_folder + 'model_best.pkl')
                args.logger.info("The path_name of best model is {} .".format(out_folder + 'model_best.pkl'))
                max_mrr = flag
            flag1, flag2, stop = early_stop(flag1, flag2, flag)
            if args.early_stop and (stop or i == args.max_epoch): # only mrr or hit@1 == 100%, stop
                print("\n == should early stop == \n")
                break
        test()
    args.logger.info("Training ends. Total time = {:.3f} s.".format(time.time() - t))


if __name__ == "__main__":
    t = time.time()
    args = read_args("config.json", True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.use_cuda else "cpu")
    out_folder = generate_out_folder(args.output, args.datasets_name, args.dataset_division, 'MSDIS')

    kg = KG(args)
    train_set = Dataset(kg.train_links, args.assessments_file)
    valid_set = Dataset(kg.valid_links, args.assessments_file)
    test_set = Dataset(kg.test_links, args.assessments_file)
    training_generator = data.DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=args.use_cuda)
    validing_generator = data.DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=args.use_cuda)
    testing_generator = data.DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=args.use_cuda)

    model = MSDIS(kg, args)
    model.to(device)
    wdl1 = Weight_Sum_Loss(6)
    wdl2 = Weight_Sum_Loss(3)
    wdl1.to(device)
    wdl2.to(device)

    optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': wdl1.parameters()}, {'params': wdl2.parameters()}], lr = args.learning_rate)
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    # scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    link_criterion = nn.MSELoss()

    train()
    # model.load_state_dict(torch.load(out_folder + 'model_best.pkl'))
    mrr = test()
    args.logger.info("Total run time = {:.3f} s.".format(time.time() - t))