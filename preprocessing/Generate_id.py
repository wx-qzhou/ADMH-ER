from utils.Path_file import read_args, join
from utils.data_utils import dump_json, load_json

"""
sort the elements based on the number of appeared elements
Input :
triples : ((h, r, t), ...), elements_set : (h, t, ...) or (r, ...)
Output :
ordered_elements : [h, t, ...] or [r, ...], dic : {h : num, t : num, ...} or {r : num, ...} 
"""
def sort_elements(triples, elements_set):
    dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in elements_set:
            dic[p] = dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1

    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]
    return ordered_elements, dic

def obtain_all_dis_entities(args):
    disambiguated_label = load_json(args.Assessment_file)
    entities_list = []
    for xxx in disambiguated_label:
        for name in disambiguated_label[xxx]:
            entities_list += disambiguated_label[xxx][name]
    return entities_list

"""
map entities and relations into ids
Input :
triples : ((h, r, t), ...), elements : (h, t, ...)
Output:
ids : {e : id}
"""
def generate_mapping_id(triples, elements, ordered=True, dis_entities_list=None):
    ids = dict()
    if ordered: # whether order?
        ordered_elements, _ = sort_elements(triples, elements)
        if dis_entities_list:
            for entity in dis_entities_list:
                ordered_elements.remove(entity)
            ordered_elements = dis_entities_list + ordered_elements
        n = len(ordered_elements)
        for i in range(n):
            ids[ordered_elements[i]] = i
    else:
        if dis_entities_list:
            for entity in dis_entities_list:
                elements.remove(entity)
            elements = dis_entities_list + elements
        index = 0
        for ele in elements:
            if ele not in ids:
                ids[ele] = index
                index += 1
    assert len(ids) == len(set(elements))
    return ids

"""
read triple data
return :
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

def generate_id(args):
    relations_triples_set, entities_set, relations_set = read_relation_triples(args.EntityTriples_file)
    dis_entities_list = obtain_all_dis_entities(args)
    dump_json(dis_entities_list, join(args.Preprocessing_data_file, "dis_entities_list.json"))
    ent_ids_dict = generate_mapping_id(relations_triples_set, entities_set, ordered=args.ordered, dis_entities_list=dis_entities_list)
    rel_ids_dict = generate_mapping_id(relations_triples_set, relations_set, ordered=args.ordered)
    dump_json(ent_ids_dict, join(args.Preprocessing_data_file, "ent_ids_dict.json"))
    dump_json(rel_ids_dict, join(args.Preprocessing_data_file, "rel_ids_dict.json"))
    pass