from utils.data_utils import load_json, dump_json, join
import time
import torch
import os

"""
extract the positive pairwise publications
Input:
assignment : [[], []]
Output:
positive_pair_list : [(id1, id2), ...]
"""
def extract_positive_pub_pair(assignment):
    positive_pair_list = []
    for group in assignment:
        for index_1, pub_id_1 in enumerate(group):
            for index_2, pub_id_2 in enumerate(group):
                if index_2 < index_1:
                    positive_pair_list.append((pub_id_1, pub_id_2))
    return positive_pair_list

"""
extract the negative pairwise publications
Input:
annotation_result : [[], [], [], [], []]
Output:
negative_pair_list : [(id1, id2), ...]
"""
def extract_negative_pub_pair(annotation_result):
    negative_pair_list = []
    for index_1, group_1 in enumerate(annotation_result): # [[], [], [], [], []]
        for index_2, group_2 in enumerate(annotation_result):
            if index_2 < index_1:
                for pub_id_1 in group_1:
                    for pub_id_2 in group_2:
                        negative_pair_list.append((pub_id_1, pub_id_2))
    return negative_pair_list

"""
save matrices
Input:
adj_list : list, assessments_file : string, file_name : string
Output:
None
"""
def save_gnn_data(
        adj_list,
        assessments_file,
        file_name,
):
    torch.save(
        adj_list,
        '{}/{}'.format(assessments_file, file_name)
    )

"""

Input:
assessment_list : [[], ...], assessments : [...]
Output:
edge_label : list
"""
def generate_edge_label(assessment_list, assessments):
    positive_pub_id_pair_list = extract_positive_pub_pair(assessment_list)
    negative_pub_id_pair_list = extract_negative_pub_pair(assessment_list)

    print('Generating edge labels...')
    label_dict = {"{}\t{}".format(pub_id_1, pub_id_2): 1 for pub_id_1, pub_id_2 in positive_pub_id_pair_list}
    label_dict.update({"{}\t{}".format(pub_id_2, pub_id_1): 1 for pub_id_1, pub_id_2 in positive_pub_id_pair_list})
    label_dict.update({"{}\t{}".format(pub_id_1, pub_id_2): 0 for pub_id_1, pub_id_2 in negative_pub_id_pair_list})
    label_dict.update({"{}\t{}".format(pub_id_2, pub_id_1): 0 for pub_id_1, pub_id_2 in negative_pub_id_pair_list})
    N = len(assessments)
    edge_label = torch.zeros((N, N)).long()
    
    for col_index in range(0, N):
        for row_index in range(0, N):
            if col_index != row_index:
                pub_id_1 = assessments[col_index]
                pub_id_2 = assessments[row_index]
                edge_label[col_index][row_index] = label_dict["{}\t{}".format(pub_id_1, pub_id_2)]
            else:
                edge_label[col_index][row_index] = 1

    return edge_label

"""
extract the case name in assessments_file
Input:
assessments_file : string
Output:
set
"""
def extract_case_name(assessments_file):
    casename_list = os.listdir(assessments_file)
    casename_list = [file_name.split(".class")[0] for file_name in casename_list]
    return set(casename_list)

def generate_labels(Assessment_file, assessments_file):
    raw_data = load_json(Assessment_file)

    casename_list = extract_case_name(assessments_file)

    for case_name in raw_data:
        if case_name not in casename_list: # 生成每个author的的处理后的数据
            print('Reading {}...'.format(case_name))
            start = time.time()
            assessment_list = []
            assessments = []
            for group_id in raw_data[case_name]:
                data = raw_data[case_name][group_id]
                assessments += data
                assessment_list.append(data)
            assessments = sorted(set(assessments), key = assessments.index)
            # print(len(assessments))
            edge_label = generate_edge_label(assessment_list, assessments)
            dump_json(assessments, join(assessments_file, case_name + ".json"))
            # print(edge_label)
            save_gnn_data(edge_label, assessments_file, case_name + ".class")
            print(time.time() - start)
        # break


if __name__ == "__main__":
    pass