from utils.data_utils import load_json, dump_json, load_json
import time
import torch
import os
from random import shuffle
from os.path import join

"""
save matrices
Input:
adj_list : list, assessments_file : string, file_name : string
Output:
None
"""
def save_gnn_data(
        adj_list,
        assessment_ids,
        assessments_file,
        file_name,
):
    torch.save(
        adj_list,
        '{}/{}'.format(assessments_file, file_name + ".link")
    )
    torch.save(
        assessment_ids,
        '{}/{}'.format(assessments_file, file_name + ".id")
    )

"""

Input:

Output:

"""
def to_id(assessment_list, assessments, entity_id_file):
    entity_id_dict = load_json(entity_id_file)
    assessment_ids = [entity_id_dict[ass] for ass in assessments]
    assessment_id_list = [[entity_id_dict[gr] for gr in group] for group in assessment_list]
    return assessment_id_list, assessment_ids

"""

Input:
assessment_list : [[], ...], assessments : [...]
Output:
edge_label : list
"""
def generate_edge_label(assessment_list, assessments, entity_id_file, sample_num):
    print('Generating ids...')
    assessment_id_list, assessment_ids = to_id(assessment_list, assessments, entity_id_file)

    assessment_keys = {}
    for index, pub in enumerate(assessment_ids):
        assessment_keys.update({pub : index})

    print('Generating edge labels...')
    samples_list = []
    for group in assessment_id_list:
        for index_1, pub_1 in enumerate(group):
            group_temp = set(group)
            negative_list = list(set(assessment_ids) - group_temp) # remove positive samples
            group_temp.remove(pub_1) # remove this sample in the set of positive samples
            for index_2, pub_2 in enumerate(group_temp): 
                shuffle(negative_list) # shuffle all negative samples
                samples_list.append([pub_1, pub_2] + sorted(negative_list[:sample_num]))

    samples_list_temp = []
    for samples in samples_list:
        samples_list_temp.append([assessment_keys[sample] for sample in samples])

    return torch.tensor(samples_list_temp).long(), torch.tensor(assessment_ids).long()

"""
extract the case name in assessments_file
Input:
assessments_file : string
Output:
set
"""
def extract_case_name(assessments_file):
    casename_list = os.listdir(assessments_file)
    casename_list = [file_name.split(".link")[0] for file_name in casename_list]
    return set(casename_list)

def generate_labels(Preprocessing_data_file, Assessment_file, assessments_file):
    raw_data = load_json(Assessment_file)

    casename_list = extract_case_name(assessments_file)

    for case_name in raw_data:
        if case_name not in casename_list: # 生成每个author的的处理后的数据
            print('Reading {}...'.format(case_name))
            start = time.time()
            assessment_list = []
            assessments_ = []
            max_len = 0
            for group_id in raw_data[case_name]:
                data = raw_data[case_name][group_id]
                max_len = max(max_len, len(data))
                assessments_ += data
                assessment_list.append(data)
            assessments_ = sorted(set(assessments_), key = assessments_.index)
            assessments = load_json(join(assessments_file, case_name + ".json"))
            assert assessments_ == assessments
            edge_label, assessment_ids = generate_edge_label(assessment_list, assessments, join(Preprocessing_data_file, "ent_ids_dict.json"), len(assessments) - max_len - 1)
            save_gnn_data(edge_label, assessment_ids, assessments_file, case_name)
            print(time.time() - start)
        # break


if __name__ == "__main__":
    assessment_list = [[1, 3, 5, 6], [2, 4, 8], [7, 9, 10, 11, 12]]
    assessments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # print(extract_positive_pub_pair(assessment_list))
    # to_id(assessment_list, assessments)
    generate_edge_label(assessment_list, assessments, 6)
    pass