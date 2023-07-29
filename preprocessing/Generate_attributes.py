from utils.data_utils import read_text, dump_json
import re
# Porter Stemmer基于Porter词干提取算法
from nltk.stem.porter import PorterStemmer  
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

sw_nltk = stopwords.words('english')
ps = PorterStemmer()  


char_in_url = ['~', '`', '!', '@', '#', '$', '%' , '^', '&', '*', '(', ')', '+', '-', '_', '=', '{', '[', '}', ']', '|', '\\', ':', ';', '"', '\'', '<', '>', '.', \
'?', ',', 'schema', 'rdf']

"""
This function is used to generate the context of attributes.
Input : 
args : {}

Output:
None
"""
def obtain_attributes(args):
    numerical_triples = read_text(args.NumericalTriples_file)
    attribute_dict = dict()
    for triple in numerical_triples:
        e, attr, v = triple
        value = attr.split(".")[-1]
        for ch in char_in_url:
            if ch in value:
                value = value.replace(ch, " ")
        pattern="[A-Z]"
        value = re.sub(pattern, lambda x : " " + x.group(0), value).strip()  
        value = value.split()
        value = [ps.stem(word) for word in value if word.lower() not in sw_nltk]
        if len(value) == 0:
            value = attr
            for ch in char_in_url:
                if ch in value:
                    value = value.replace(ch, " ")
            pattern="[A-Z]"
            value = re.sub(pattern, lambda x : " " + x.group(0), value).strip()  
            value = value.split()
            value = [ps.stem(word) for word in value if word.lower() not in sw_nltk]
        if len(value) == 0:
            value = attr.split(" ")

        pattern="[-.]"
        v_ = re.sub(pattern, lambda x : " " + x.group(0) + " ", str(v)).strip()
        attribute_dict.update({attr : " ".join(sorted(set(value), key=value.index))})
        attribute_dict.update({str(v) : v_})
    dump_json(attribute_dict, args.AttributeText_file)

"""
This function is to generate the n-gramma of attributes.
Input : 
args : {}, n : number

Output:
None
"""
def obtain_n_gramma(args, n):
    numerical_triples = read_text(args.NumericalTriples_file)
    attribute_dict = dict()
    for triple in numerical_triples:
        e, attr, v = triple
        value = attr.split(".")[-1]
        for ch in char_in_url:
            if ch in value:
                value = value.replace(ch, " ")
        pattern="[A-Z]"
        value = re.sub(pattern, lambda x : " " + x.group(0), value).strip()  
        value = value.split()
        value = [ps.stem(word) for word in value if word.lower() not in sw_nltk]
        if len(value) == 0:
            value = attr
            for ch in char_in_url:
                if ch in value:
                    value = value.replace(ch, " ")
            pattern="[A-Z]"
            value = re.sub(pattern, lambda x : " " + x.group(0), value).strip()  
            value = value.split()
            value = [ps.stem(word) for word in value if word.lower() not in sw_nltk]        
        if len(value) == 0:
            value = attr.split(" ")
                    
        pattern="[-.]"
        v_ = re.sub(pattern, lambda x : " " + x.group(0) + " ", str(v)).strip()

        value = sorted(set(value), key=value.index)
        char_list = []
        for val in value:
            if len(val) > n:
                i = n
                while i < len(val):
                    char_list.append(val[i-n : i])
                    i += n
                if i - n != len(val):
                    char_list.append(val[i-n : len(val)])
            else:
                char_list.append(val)

        v_ = v_.split()
        v_char_list = []
        for val in v_:
            if len(val) > n:
                i = n
                while i < len(val):
                    v_char_list.append(val[i-n : i])
                    i += n
                if i - n != len(val):
                    v_char_list.append(val[i-n : len(val)])
            else:
                v_char_list.append(val)

        attribute_dict.update({attr : " ".join(char_list)})
        attribute_dict.update({str(v) : " ".join(v_char_list)})
    dump_json(attribute_dict, args.AttributeNGram_file)

def generate_entity_attributes(args):
    numerical_triples = read_text(args.NumericalTriples_file)
    entity_attribute_dict = dict()
    for triple in numerical_triples:
        e, attr, v = triple
        value = attr
        for ch in char_in_url:
            if ch in value:
                value = value.replace(ch, " ")
        pattern="[A-Z]"
        value = re.sub(pattern, lambda x : " " + x.group(0), value).strip()  

        pattern="[-.]"
        v_ = re.sub(pattern, lambda x : " " + x.group(0) + " ", str(v)).strip()
        value = value + " " + str(v_)
        entity_attribute_dict.update({e : sorted(set(value.split()), key=value.split().index)})
    dump_json(entity_attribute_dict, join(args.Preprocessing_text_file, "entity_attribute_txt.json"))
