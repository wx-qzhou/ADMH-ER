import codecs
import json
import numpy as np
from os.path import join
import pickle
import os
import jieba
import warnings
warnings.filterwarnings("ignore")

# read the list of stopwords
def get_stopword_list(file):
    with open(file, 'r', encoding='utf-8') as f:    # 
        stopword_list = [word.strip('\n') for word in f.readlines()]
    return list(set(stopword_list))

# participle and then clear stopwords
def clean_stopword(str, stopword_list):
    result = []
    word_list = jieba.lcut(str)   # 分词后返回一个列表  jieba.cut(）   返回的是一个迭代器
    for w in word_list:
        if w not in stopword_list:
            result.append(w)
    return " ".join(result)
    
# loads make str to dict
def load_json(rffile):
    with codecs.open(rffile, 'r', encoding='utf-8') as rf:
        return json.load(rf)

# dumps makes dict to str
def dump_json(obj, wffile, indent=None):
    with codecs.open(wffile, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)

# write binary file
def dump_data(obj, wffile):
    with open(wffile, 'wb') as wf:
        pickle.dump(obj, wf)

# read binary file
def load_data(rffile):
    with open(rffile, 'rb') as rf:
        return pickle.load(rf)

# string to json
def serialize_embedding(embedding):
    return pickle.dumps(embedding)

# json to string
def deserialize_embedding(s):
    return pickle.loads(s)

def wirte_txt(file_name, data):
    with open(file_name,'w') as f:
        for dat in data:
            f.write(str(dat[0]) + "\t")
            f.write(str(dat[1]) + "\t")
            f.write(str(dat[2]))
            f.write("\n")
        f.close()

"""
# read triples
Input : 
file_name : string

Output:
np.array(triple_list) : np.array([[h, r, t], ...])
"""
def read_text(file_name):
    triple_list = []
    with open(file_name, "r") as f:
        for triple in list(f.readlines()):
            triple = triple.strip("\n").strip().split("\t")
            triple_list.append(triple)

    return np.array(triple_list)