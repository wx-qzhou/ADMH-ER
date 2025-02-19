# ADMH-ER
ADMH-ER

## Descriptions

Firstly, we try to generate all modal data based on pre-trained models.
1) Text : we obtain the form of representations, i.e., {"name" : representation}
2) Images : we obtain the form of representations, i.e., {"name" : representation}
3) Videos : we obtain the form of representations, i.e., {"name" : representation}

Secondly, we generate the ids of entities and relations.
entities : {entities' name : id}
relations : {relations' name : id}

Thirdly, we generate the classification labels and linking labels.
Note that, we create the groundtruth with the names.
For each name, we have four files, i.e., "xxx.class", "xxx.json" and "xxx.link"

---

## Datasets

The raw datasets can be found below.
### 1. DBP15k Datasets

It is available at the following GitHub repository:
- [EVA on GitHub](https://github.com/cambridgeltl/eva)


### 2. MMKB Datasets

It can be accessed via:
- [MMKG Dataset on GitHub](https://github.com/mniepert/mmkb)

Our processing Datasets for MMER can be found below.
### 1. MMER Datasets
The rough datasets can be available, and more details of them are coming. You can download them from:
- [MMER Datasets on BaiduPan](Link: https://pan.baidu.com/s/1kbUPWreOKUcrwaBXNDjJFA?pwd=heb9 password: heb9)

## Citation
```bibtex
@article{Zhou2025ADMHERAD,
  title={ADMH-ER: Adaptive Denoising Multi-Modal Hybrid for Entity Resolution},
  author={Qian Zhou and Wei Chen and Li Zhang and An Liu and Jun Fang and Lei Zhao},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025},
  volume={37},
  pages={1049-1063},
  url={https://api.semanticscholar.org/CorpusID:275448400}
}
