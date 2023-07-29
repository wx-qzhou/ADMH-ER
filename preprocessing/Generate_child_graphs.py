from os.path import join
from utils.data_utils import load_json
import os
import torch
"""
# read triple data

Input :
file_path : string
Output :
triples : ((h, r, t), ...), entities : (h, t, ...), relations : (r, ...)
"""
def read_relation_triples(file_path):
    print("read relation triples:", file_path)
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    file.close()
    return triples, entities, relations

"""
# read the ids of entities and relations

Input :
file_path : string
Output:
ids : {e : id}, {r : id}
"""
def read_mapping_id(file_path):
    ent_ids = load_json(join(file_path, "ent_ids_dict.json"))
    rel_ids = load_json(join(file_path, "rel_ids_dict.json"))
    return ent_ids, rel_ids

def generate_childgraphs(raw_child_graphs, dis_entities_ids):
    h_rt = {}
    
    for ent_id in dis_entities_ids:
        pass
    pass

def main_child_graphs(datasets_file, EntityTriples_file, assessments_file):
    ent_ids, rel_ids = read_mapping_id(datasets_file)
    triples, _, _ = read_relation_triples(EntityTriples_file)
    dis_name = set()
    for file_name in os.listdir(assessments_file):
        dis_name.add(file_name.split(".")[0])
    for file_name in dis_name:
        raw_child_graphs = list()
        dis_entities_list = list(load_json(join(assessments_file, file_name + ".json")))
        dis_entities_ids = set()
        for h, r, t in triples:
            if h in dis_entities_list:
                raw_child_graphs.append((ent_ids[h], rel_ids[r], ent_ids[t]))
                dis_entities_ids.add(ent_ids[h])
            if t in dis_entities_list:
                raw_child_graphs.append((ent_ids[h], rel_ids[r], ent_ids[t]))
                dis_entities_ids.add(ent_ids[t])
        torch.save(
            torch.LongTensor(raw_child_graphs),
            '{}/{}'.format(assessments_file, file_name + ".triples")
        )