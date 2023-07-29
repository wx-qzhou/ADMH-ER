from os.path import abspath, dirname, join, exists
from os import mkdir, makedirs
import time
import logging
from utils.data_utils import load_json

def make_dir(path):
    if not exists(path):
        mkdir(path)

def make_dirs(path):
    if not exists(path):
        makedirs(path)

class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

def generate_out_folder(out_folder, datasets_name, div_path, method_name):
    """
    # generate the dir of output
    Input:
    out_folder : string, datasets_name : string, div_path, method_name : string

    Output:
    folder : string
    """
    folder = join(out_folder, datasets_name, method_name + div_path)
    print("results output folder:", folder)
    make_dirs(folder)
    return folder + "/" + str(time.strftime("%Y%m%d%H%M%S"))

def set_logger(project_name, datasets_name):
    filename = join(project_name, datasets_name, 'logs')
    make_dirs(filename)
    filename = filename + "/" + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log'
    logger = logging.getLogger(filename)
    format_str = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = logging.FileHandler(filename, mode='a', encoding='utf-8', delay=False)
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger

def main_args(args_dict):
    # the path of entity triples
    EntityTriples_file = join(args_dict["datasets_file"], "Triples", "entities_triples")
    # the path of numerical triples
    NumericalTriples_file = join(args_dict["datasets_file"], "Triples", "numerical_triples")
    # the path of image representations
    Image_Representation_file = join(args_dict["Preprocessing_H5_file"], "Image_Representation.h5")
    # the path of text representations
    Text_Representation_file = join(args_dict["Preprocessing_H5_file"], "Text_Representation.h5")
    # the path of attribute representations
    Attribute_Representation_file = join(args_dict["Preprocessing_H5_file"], "Attribute_Representation.h5")
    AttributeNGram_Representation_file = join(args_dict["Preprocessing_H5_file"], "AttributeNGram_Representation.h5")

    args_dict.update({"EntityTriples_file" : EntityTriples_file})
    args_dict.update({"NumericalTriples_file" : NumericalTriples_file})
    args_dict.update({"Image_Representation_file" : Image_Representation_file})
    args_dict.update({"Text_Representation_file" : Text_Representation_file})
    args_dict.update({"Attribute_Representation_file" : Attribute_Representation_file})
    args_dict.update({"AttributeNGram_Representation_file" : AttributeNGram_Representation_file})
    
    return args_dict

def preprocess_args(args_dict):
    # create new folders
    Preprocessing_data_file = join(args_dict["datasets_file"], "Preprocessing_data")
    make_dir(Preprocessing_data_file)
    assessments_file = join(Preprocessing_data_file, "assessments")
    make_dir(assessments_file)
    Preprocessing_text_file = join(Preprocessing_data_file, "text")
    make_dir(Preprocessing_text_file)
    Preprocessing_H5_file = join(Preprocessing_data_file, "embH5")
    make_dir(Preprocessing_H5_file)
    
    args_dict.update({"Preprocessing_data_file" : Preprocessing_data_file})   
    args_dict.update({"assessments_file" : assessments_file})
    args_dict.update({"Preprocessing_text_file" : Preprocessing_text_file})
    args_dict.update({"Preprocessing_H5_file" : Preprocessing_H5_file})

    args_dict.update({"Feature_Idf_file" : join(Preprocessing_H5_file, "feature_idf.pkl")})
    args_dict.update({"ImageIndex_file" : join(args_dict["datasets_file"], "Images", "image_index.txt")})
    args_dict.update({"EntityText_file" : join(args_dict["datasets_file"], "Text", "entities_txt.json")})
    args_dict.update({"AttributeText_file" : join(Preprocessing_text_file, "attribute_txt.json")})
    args_dict.update({"AttributeNGram_file" : join(Preprocessing_text_file, "attribute_n_gram_txt.json")})
    args_dict.update({"RelationText_file" : join(Preprocessing_text_file, "relation_txt.json")})

    return args_dict

def read_args(args_name, is_logger=False):
    # the path of main project
    project_name = dirname(dirname(abspath(__file__)))

    # download the settings in json. 
    args_dict = load_json(join(project_name, args_name))

    # the path of dataset
    args_dict.update({"datasets_file" : join(project_name, "datasets", args_dict["datasets_name"])})
    args_dict.update({"Assessment_file" : join(project_name, "datasets", args_dict["datasets_name"], "assessment.json")})

    # the path of pretrained models
    args_dict.update({"Pretrained_models_file" : join(project_name, "representations", "Pretrained_models")})

    # folder of output
    args_dict.update({"output" : join(project_name, "result")})
    
    args_dict = preprocess_args(args_dict)

    args_dict = main_args(args_dict)

    # whether the code needs to set log or not
    if is_logger:
        logger = set_logger(args_dict["output"], args_dict["datasets_name"])
        args_dict.update({"logger" : logger})
        for (k, v) in args_dict.items():
            logger.info(str(k)+' : '+str(v))

    args = ARGs(args_dict)
    return args