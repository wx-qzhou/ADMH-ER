from utils.Path_file import *
from preprocessing.Generate_id import generate_id
from preprocessing.Generate_child_graphs import main_child_graphs
from preprocessing.Generate_adj_each_name import main_child_graph_adj

def Pretrained_Text_model(args, model):
    """
    to obtain the representations of text.
    Input :
    args : dict, model : string
    Output :
    None
    """
    print("The pretrained model is {}.".format(model))
    if model == "Bert":
        from representations.Text.EmbeddingwithBert import Bert_main
        Bert_main(args.EntityText_file, args.Text_Representation_file, args.Pretrained_models_file)
    if model == "Glove":
        from representations.Text.EmbeddingwithGlove import Glove_main
        Glove_main(args.EntityText_file, args.Text_Representation_file, args.Pretrained_models_file, no_threads=10, is_training=True)
    if model == "LABSE":
        from representations.Text.EmbeddingwithLABSE import LABSE_main
        LABSE_main(args.EntityText_file, args.Text_Representation_file, gpuid="1")
    if model == "IDF":
        from representations.Text.Feature_idf import IDF_Triples
        IDF_Triples(args.EntityTriples_file, args.NumericalTriples_file, args.Feature_Idf_file)

def Pretrained_Image_model(args, model):
    """
    to obtain the representations of images.
    Input :
    args : dict, model : string
    Output :
    None
    """
    print("The pretrained model is {}.".format(model))
    if model == "Alexnet":
        from representations.Image.Images_Alexnet import Image_Alexnet_main
        Image_Alexnet_main(args.datasets_file, args.Image_Representation_file)
    elif model == "VIT":
        from representations.Image.Images_ViT import Image_ViT_main
        Image_ViT_main(args.datasets_file, args.Image_Representation_file, args.Pretrained_models_file)

def Pretrained_Attributes_model(args, model, isObtianAttributes, isObtianAttributes_n_gram, n=2):
    """
    to obtain the representations of images.
    Input :
    args : dict, model : string
    Output :
    None
    """
    attributeText_file = args.AttributeText_file
    attribute_Representation_file = args.Attribute_Representation_file
    if isObtianAttributes:
        from preprocessing.Generate_attributes import obtain_attributes
        obtain_attributes(args)
        attributeText_file = args.AttributeText_file
        attribute_Representation_file = args.Attribute_Representation_file
    elif isObtianAttributes_n_gram:
        from preprocessing.Generate_attributes import obtain_n_gramma
        obtain_n_gramma(args, n)
        attributeText_file = args.AttributeNGram_file
        attribute_Representation_file = args.AttributeNGram_Representation_file

    print("The pretrained model is {}.".format(model))
    if model == "Word2Vec":
        from representations.Text.EmbeddingwithWord2Vec import Word2Vec_main
        Word2Vec_main(attributeText_file, attribute_Representation_file, args.Feature_Idf_file)
    if model == "Doc2Vec":
        from representations.Text.EmbeddingwithDoc2Vec import Doc2Vec_main
        Doc2Vec_main(attributeText_file, attribute_Representation_file)
    if model == "Glove":
        from representations.Text.EmbeddingwithGlove import Glove_main
        Glove_main(attributeText_file, attribute_Representation_file, args.Pretrained_models_file, no_threads=10, is_training=True)
    if model == "LABSE":
        from representations.Text.EmbeddingwithLABSE import LABSE_main
        LABSE_main(attributeText_file, attribute_Representation_file, gpuid="1")

def Obtain_labels(args, classification=False):
    if classification:
        from preprocessing.Generate_classification_label import generate_labels
        generate_labels(args.Assessment_file, args.assessments_file)
    else:
        from preprocessing.Generate_label import generate_labels
        generate_labels(args.Preprocessing_data_file, args.Assessment_file, args.assessments_file)


if __name__ == "__main__":
    args = read_args("config.json")
    print(args.datasets_name)
    # print("Firstly, we obtain the representations of modal data.")
    # # Pretrained_Attributes_model(args, "Glove", False, True)
    # # Pretrained_Attributes_model(args, "Word2Vec", True, False)
    # Pretrained_Attributes_model(args, "LABSE", True, False)
    # # Pretrained_Attributes_model(args, "Doc2Vec", False, True)
    # # Pretrained_Text_model(args, "Bert")
    # # Pretrained_Text_model(args, "Glove")
    # Pretrained_Text_model(args, "IDF")
    # Pretrained_Text_model(args, "LABSE")
    # Pretrained_Image_model(args, "VIT")
    # # Pretrained_Image_model(args, "Alexnet")
    # print("Secondly, we generate the ids of entities and relations.")
    # generate_id(args)
    # print("Thirdly, we generate the ids of entities and relations.")
    # Obtain_labels(args, True)
    # Obtain_labels(args)
    # main_child_graphs(args.Preprocessing_data_file, args.EntityTriples_file, args.assessments_file)
    main_child_graph_adj(args.Preprocessing_data_file, args.assessments_file)
