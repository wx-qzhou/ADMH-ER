import json
from os.path import join

def read_json(file_name="./Preprocessing_data/ent_ids_dict.json"):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, file_name, file_dir="./Preprocessing_data/"):
    with open(join(file_dir, file_name), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_triple(file_name="./Triples/entities_triples"):
    with open(file_name, 'r', encoding='utf-8') as file:
        triples = []
        for line in file:
            # 将每一行拆分成三元组
            subject, predicate, object = line.strip().split("\t")
            triples.append((subject, predicate, object))
    return triples

def write_triples(triples, triples_file):
    with open(triples_file, 'w', encoding='utf-8') as file:
        for subject, predicate, object in triples:
            file.write(f"{subject} {predicate} {object}\n")

def ID_change_triple(ent_id, rel_id, triple):
    ID_triple = []
    for h, r, t in triple:
        ID_triple.append((ent_id[h], rel_id[r], ent_id[t]))
    return ID_triple

def ID_change_numtriple(ent_id, triple):
    ID_triple = []
    attr_dict = dict()
    value_dict = dict()
    ai, vi = 0, 0
    for e, a, v in triple:
        if a not in attr_dict:
            attr_dict[a] = ai
            ai = ai + 1
        if v not in value_dict:
            value_dict[v] = vi
            vi = vi + 1
        ID_triple.append((ent_id[e], attr_dict[a], value_dict[v]))
    return ID_triple, attr_dict, value_dict

ent_id = read_json()
rel_id = read_json("./Preprocessing_data/rel_ids_dict.json")
triple = read_triple()
num_triple = read_triple("./Triples/numerical_triples")
triple = ID_change_triple(ent_id, rel_id, triple)
num_triple, attr_dict, value_dict = ID_change_numtriple(ent_id, num_triple)
triple = sorted(triple, key=lambda x: x[0])
num_triple = sorted(num_triple, key=lambda x: x[0])

write_triples(triple, "./ID_entities_triples")
write_triples(num_triple, "./ID_numerical_triples")

save_json(attr_dict, "attr_ids_dict.json")
save_json(value_dict, "values_ids_dict.json")