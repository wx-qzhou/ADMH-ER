import h5py
from os.path import join
import numpy as np
import torch
from utils.data_utils import load_json, load_data
from collections import Counter

np.set_printoptions(threshold=np.inf)

class KG:
    def __init__(self, args):
        self.args = args
        # The following step is to read the triples, read the ids of entites and relations, and transform the raw triples into the id triples.
        relations_triples_set, entities_set, relations_set, entity_re_dict = self.read_relation_triples(args.EntityTriples_file)
        self.ent_num, self.relation_triples_num, self.rel_num = len(entities_set), len(relations_triples_set), len(relations_set)
        print("The numbers of entities and relations are {} and {}.".format(self.ent_num, self.rel_num))
        del entities_set, relations_set

        self.ent_ids_dict, self.rel_ids_dict = self.read_mapping_id(args.Preprocessing_data_file)
        self.id_relation_triples, rt_dict, hr_dict = self.uris_relation_triple_2ids(relations_triples_set)
        del relations_triples_set

        e_re, e_max_num, r_max_num, self.entid_rel_list, self.entid_ent_list, self.e_mask, self.r_mask = self.generate_walk(rt_dict, hr_dict)
        self.generate_n_htop(3, e_re)
        del e_re, rt_dict, hr_dict

        # The following step is to split the data based on names.
        self.train_links, self.valid_links, self.test_links = self.split_data(args.Assessment_file, args.dataset_division)
        # self.train_links, self.valid_links, self.test_links = self.train_links, self.train_links, self.train_links 
        # self.train_links, self.valid_links, self.test_links = ['xxx0'], ['xxx0'], ['xxx0']
        self.dis_entites = load_json(join(args.Preprocessing_data_file, "dis_entities_list.json"))
        self.dis_entities_len = len(self.dis_entites)
        
        # Next, obtain the image ids and read the representations of images.
        self.id_ent_images_res_dict, self.images_list = self.read_image_embeddings(args.Image_Representation_file, self.read_image_ID(args.ImageIndex_file), args.i_dim)

        # 
        entity_av_dict = self.read_attribues(args.NumericalTriples_file)
        
        # Afterward, read embeddings of attributes.
        # self.attr_list = self.read_attr_embeddings(entity_av_dict, args.Attribute_Representation_file, args.a_dim)
        self.attr_list = self.load_attr(entity_av_dict, entity_re_dict, args.Feature_Idf_file, topA=768)
        del entity_av_dict, entity_re_dict

        # Then we try to read embeddings of text.
        self.txt_embed, self.txt_list = self.read_txt_embeddings(args.Text_Representation_file, args.t_dim)


    def read_relation_triples(self, file_path):
        """
        # read triple data

        Input :
        file_path : string
        Output :
        triples : ((h, r, t), ...), entities : (h, t, ...), relations : (r, ...)
        """
        print("read relation triples:", file_path)
        triples = set()
        entities, relations = set(), set()
        entity_re_dict = {}
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

            rt_set = entity_re_dict.get(h, set())
            rt_set.add((r, t))
            entity_re_dict[h] = rt_set
            rh_set = entity_re_dict.get(t, set())
            rh_set.add((r, h))
            entity_re_dict[t] = rh_set
        file.close()
        return triples, entities, relations, entity_re_dict

    def read_mapping_id(self, file_path):
        """
        # read the ids of entities and relations

        Input :
        file_path : string
        Output:
        ids : {e : id}, {r : id}
        """
        ent_ids = load_json(join(file_path, "ent_ids_dict.json"))
        rel_ids = load_json(join(file_path, "rel_ids_dict.json"))
        return ent_ids, rel_ids

    def uris_relation_triple_2ids(self, uris):
        """
        # make all string to id and generate {hid : (rid, tid)} and {tid : (hid, rid)}

        Input : 
        uris : ((h, r, t), ...)
        Output :
        id_uris : [(h_id, r_id, t_id), ...], rt_dict : {h_id : {(r_id, t_id), ...}, ...}, hr_dict : {t_id : {(r_id, h_id), ...}, ...}
        """
        id_uris = list()
        rt_dict, hr_dict = dict(), dict()
        for u1, u2, u3 in uris:
            assert u1 in self.ent_ids_dict
            h_id = self.ent_ids_dict[u1]
            assert u2 in self.rel_ids_dict
            r_id = self.rel_ids_dict[u2]
            assert u3 in self.ent_ids_dict
            t_id = self.ent_ids_dict[u3]
            id_uris.append((h_id, r_id, t_id))

            rt_set = rt_dict.get(h_id, set())
            rt_set.add((r_id, t_id))
            rt_dict[h_id] = rt_set

            hr_set = hr_dict.get(t_id, set())
            hr_set.add((r_id, h_id))
            hr_dict[t_id] = hr_set

        assert len(id_uris) == len(set(uris))
        return id_uris, rt_dict, hr_dict

    def generate_walk(self, rt_dict, hr_dict):
        """
        # generate one htop neighborhoods

        Input : 
        rt_dict : {h_id : {(r_id, t_id), ...}, ...}, hr_dict : {t_id : {(r_id, h_id), ...}, ...}
        Output :
        e_re : {e_id : {(r_id, e_id), ...}, ...}
        """
        e_re = {} # join data : {(r, e)}
        for e in range(self.ent_num):
            ht_list = []
            try:
                ht_list = list(hr_dict[e])
            except:
                pass
            try:
                ht_list += list(rt_dict[e])
            except:
                pass
            ht_list = set(ht_list)
            e_re.update({e : ht_list})

        # e_len_num = []
        # r_len_num = []
        er_dict = dict() # {e : [[e, ...], [r, ...]]}
        for key in e_re:
            e_set = set()
            r_set = set()
            for value in e_re[key]:
                e_set.add(value[1])
                r_set.add(value[0])
            # e_len_num.append(len(e_set))
            # r_len_num.append(len(r_set))
            er_dict[key] = [e_set, r_set]

        e_list = [] # [e, ...]
        r_list = [] # [r, ...]
        for key in er_dict:
            e_list += list(er_dict[key][0])
            r_list += list(er_dict[key][1])
        e_list = Counter(e_list)
        e_list = set([key for key in e_list if e_list[key] >= 1]) # clear the data that is no interactions
        r_list = Counter(r_list)
        r_list = set([key for key in r_list if r_list[key] >= 1]) # clear the data that is no interactions

        e_len_num = []
        r_len_num = []
        for key in er_dict:
            e_temp = er_dict[key][0].intersection(e_list)
            r_temp = er_dict[key][1].intersection(r_list)
            if len(e_temp) != 0:
                er_dict[key][0] = e_temp
            if len(r_temp) != 0:
                er_dict[key][1] = r_temp
            er_dict[key][0] = sorted(er_dict[key][0])
            er_dict[key][1] = sorted(er_dict[key][1])
            e_len_num.append(len(er_dict[key][0]))
            r_len_num.append(len(er_dict[key][1]))

        e_max_num = max(e_len_num)
        r_max_num = max(r_len_num)
        print("The max length of random walks' e is {}".format(e_max_num))
        print("The max length of random walks' r is {}".format(r_max_num))

        entid_rel_list = []
        entid_ent_list = []
        e_mask = np.ones((self.ent_num, e_max_num))
        r_mask = np.ones((self.ent_num, r_max_num))
        for eid in range(self.ent_num):
            try:
                e_l = len(er_dict[eid][0])
                r_l = len(er_dict[eid][1])
                re = er_dict[eid] # [{a_id, ...}, {v, ...}]
            except Exception as ex:
                e_l = 0
                r_l = 0
                re = [set(), set()]
            e = list(re[0])
            r = list(re[1])
            for i in range(e_max_num - e_l):
                e.append(0)
            for i in range(r_max_num - r_l):
                r.append(0)
            e_mask[eid][e_l:] = 0
            r_mask[eid][r_l:] = 0
            entid_ent_list.append(e[:e_max_num])
            entid_rel_list.append(r[:r_max_num])
        del er_dict
        return e_re, e_max_num, r_max_num, torch.LongTensor(entid_rel_list), torch.LongTensor(entid_ent_list), torch.LongTensor(e_mask.tolist()), torch.LongTensor(r_mask.tolist())

    def generate_n_htop(self, htop, h_rt_dict):
        """
        # generate n htop neighborhoods

        Input : 
        htop : {e_id : {(r_id, e_id), ...}, ...}, h_rt_dict
        Output :
        None
        """
        self.mutil_htop = {}
        for h in range(self.ent_num):
            # mutil-hop nodes
            i = 1
            rt_list = []
            try:
                rt_list.append(h_rt_dict[h])
                while i < htop:
                    tmp = []
                    for r, t in rt_list[i]:
                        try:
                            tmp += h_rt_dict[t]
                        except:
                            pass
                    if len(tmp) == 0:
                        break
                    tmp = list(set(tmp))
                    rt_list.append(tmp)
                    i += 1
            except:
                pass


            e_set = set()
            for rt_ in rt_list:
                for rt in rt_:
                    e_set.add(rt[1])

            if h not in self.mutil_htop:
                self.mutil_htop.update({h : e_set})
            else:
                self.mutil_htop[h] |= e_set
    
    def read_image_ID(self, file_path):
        """
        # obtain the iamge ID

        Input: 
        file_path : string
        Output:
        links : {e : Image_id, ...}
        """
        print("read image_ID:", file_path)
        links = dict()
        file = open(file_path, 'r', encoding='utf8')
        for line in file.readlines():
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            links.update({params[0] : params[1]})
        return links

    def read_image_embeddings(self, h5_file, image_ID, i_dim=2048):
        """
        # read the embeddings of images

        Input:
        h5_file : string, image_ID : {e : Image_id, ...}, i_dim : int
        Output:
        entid_image : {id : embeddings}, images_list : [embeddings]
        """
        print("read image embeddings:", h5_file)
        entid_image = dict()
        f = h5py.File(h5_file, 'r')
        index = 0
        for (ent, id) in self.ent_ids_dict.items():
            try:
                entid_image[id] = np.array(f[image_ID[ent]]).reshape(-1)
            except:
                if ent in self.dis_entites:
                    index += 1
                    entid_image[id] = np.zeros(i_dim)
        print("No image embeddings' number is {}".format(index))
        f.close()
        del image_ID
        images_list = []
        
        for id in sorted(entid_image.keys()):
            images_list.append(entid_image[id])
        return entid_image, images_list

    def split_data(self, label_file, division):
        """
        # split data into train, test, and valid

        Input:
        label_file : string, division : string
        Output:
        train : [], valid : [], test : []
        """
        disambiguated_label = load_json(label_file)
        division = list(map(int, list(division)))
        links_data = list(disambiguated_label.keys())
        del disambiguated_label
        data_len = int(len(links_data) / (division[0] + division[1] + division[2]))
        return links_data[ : division[0] * data_len], links_data[division[0] * data_len : \
            (division[0] + division[1]) * data_len], \
            links_data[(division[0] + division[1]) * data_len : ]

    def read_attribues(self, att_f1):
        """
        # read the data about attribute triples.

        Input :
        att_f1 : str
        Output :
        entity_av_dict : {e : {(a, v)}}
        """
        entity_av_dict = {}
        file = open(att_f1, 'r', encoding='utf-8')
        for line in file.readlines():
            params = line.strip().split('\t')
            assert len(params) == 3
            e = params[0].strip()
            a = params[1].strip()
            v = params[2].strip()

            'deal with the values'
            v = v.split('\"^^')[0].strip('\"')
            if 'e-' in v:
                pass
            elif '-' in v and v[0] != '-':
                v = v.split('-')[0]
            elif v[0] == '-' and v.count('-') > 1:
                v = '-' + v.split('-')[1]
            if '#' in v:
                v = v.strip('#')

            av_set = entity_av_dict.get(e, set())
            av_set.add((a, float(v)))
            entity_av_dict[e] = av_set
        file.close()
        return entity_av_dict

    def read_attr_embeddings(self, entity_av_dict, h5_file, a_dim=100):
        """
        # read the embeddings of attributes.

        Input:
        entity_av_dict : {e:(a, v)}, h5_file : string, a_dim : int
        Output:
        attr_embed : {id : embeddings}, attr_list : [embedding1, ...]
        """
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            return s

        attr_embed = dict()
        f = h5py.File(h5_file, 'r')
        index = 0
        for (ent, id) in self.ent_ids_dict.items():
            try:
                temp_list = []
                for av in entity_av_dict[ent]:
                    attr, value = av
                    temp_list.append(np.array(f[attr]))
                    temp_list.append(np.array(f[str(value)]))
                temp_list = np.array(temp_list)
                temp_list = np.mean(temp_list, 0)
                attr_embed[id] = temp_list.reshape(-1)
            except:
                if ent in self.dis_entites:
                    index += 1
                    attr_embed[id] = np.zeros(a_dim)
        print("No attribute embeddings' number is {}".format(index))

        attr_list = []
        for id in sorted(attr_embed.keys()):
            attr_list.append(attr_embed[id])

        return attr_list

    def read_txt_embeddings(self, h5_file, t_dim=100):
        """
        # read the embeddings of text

        Input:
        h5_file : string, t_dim : int
        Output:
        txt_embed : {id : embeddings}, txt_list : [embedding1, ...]
        """
        txt_embed = dict()

        f = h5py.File(h5_file, 'r')
        index = 0
        for (ent, id) in self.ent_ids_dict.items():
            try:
                txt_embed[id] = np.array(f[ent]).reshape(-1)
            except:
                if ent in self.dis_entites:
                    index += 1
                    txt_embed[id] = np.zeros(t_dim)
        print("No text embeddings' number is {}".format(index))
        txt_list = []

        # exit()
        
        for id in sorted(txt_embed.keys()):
            txt_list.append(txt_embed[id])
        return txt_embed, txt_list

    def load_attr(self, entity_av_dict, entity_re_dict, Feature_Idf_file, topA=768):
        """
        # read the frequency of attr and rel 

        Input:
        entity_av_dict : {e : {(a, v)}}, entity_re_dict : {e : {(r, e)}}
        Output:
        txt_embed : {id : embeddings}, txt_list : [embedding1, ...]
        """
        cnt = {}
        for e in entity_re_dict:
            if e not in self.dis_entites:
                continue
            th = list(entity_re_dict[e])
            for i in range(0, len(th)):
                if th[i][0] not in cnt:
                    cnt[th[i][0]] = 1
                else:
                    cnt[th[i][0]] += 1

        for e in entity_av_dict:
            if e not in self.dis_entites:
                continue
            th = list(entity_av_dict[e])
            for i in range(0, len(th)):
                if th[i][0] not in cnt:
                    cnt[th[i][0]] = 1
                else:
                    cnt[th[i][0]] += 1
                if th[i][1] not in cnt:
                    cnt[th[i][1]] = 1
                else:
                    cnt[th[i][1]] += 1
        fre = [(k, cnt[k]) for k in dict(sorted(cnt.items(), key=lambda x:x[1], reverse=True))] # the frequency of attributes

        attr2id = {}
        for i in range(min(topA, len(fre))):
            attr2id[fre[i][0]] = i

        idf = load_data(Feature_Idf_file)
        idf_mean = np.mean(list(idf.values()))
        attr = np.zeros((len(self.dis_entites), topA), dtype=np.float32)

        for e in entity_re_dict:
            if e not in self.dis_entites:
                continue
            th = list(entity_re_dict[e])
            for i in range(0, len(th)):
                if th[i][0] in attr2id:
                    try:
                        attr[self.ent_ids_dict[e]][attr2id[th[i][0]]] = cnt[th[i][0]] * idf[th[i][0]]
                    except:
                        attr[self.ent_ids_dict[e]][attr2id[th[i][0]]] = cnt[th[i][0]] * idf_mean

        for e in entity_av_dict:
            if e not in self.dis_entites:
                continue
            th = list(entity_av_dict[e])
            for i in range(0, len(th)):
                if th[i][0] in attr2id:
                    try:
                        attr[self.ent_ids_dict[e]][attr2id[th[i][0]]] = cnt[th[i][0]] * idf[th[i][0]]
                    except:
                        attr[self.ent_ids_dict[e]][attr2id[th[i][0]]] = cnt[th[i][0]] * idf_mean
                if th[i][1] in attr2id:
                    try:
                        attr[self.ent_ids_dict[e]][attr2id[th[i][1]]] = cnt[th[i][1]] * idf[th[i][1]]
                    except:
                        attr[self.ent_ids_dict[e]][attr2id[th[i][1]]] = cnt[th[i][1]] * idf_mean

        return attr # [[1, 0, ...] * e]