# Selection Bias Explorations and Debias Methods for Natural Language Sentence Matching Datasets

This is the code in [Selection Bias Explorations and Debias Methods for Natural Language Sentence Matching Datasets](<https://arxiv.org/abs/1905.06221>) which has been accepted by ACL 2019.

## Folders

​	*<u>quantify</u>* contains codes for generating weights and codes for *Section 2.1 Quantifying the Biasedness in Datasets* in which we explore the severity of the leakage in six NLSM datasest.

​	*<u>debias</u>* contains codes for *Section 5 Experimental Results for the Leakage-neutral Method on QuoraQP* where we apply our leakage-neutral learning in QuoraQP with a classical Siamese-LSTM model.

​	**Usage and requirements are stated inside folders.**

## Datasets

We use following six datasets in our paper:

- [QuoraQP](<https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view>)
- [MSRP](<https://www.microsoft.com/en-us/download/details.aspx?id=52398>)
- [SICK](<http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools>)
- [SNLI](<https://nlp.stanford.edu/projects/snli/>)
- [MultiNLI](<https://www.nyu.edu/projects/bowman/multinli/>)
- [ByteDance](<https://www.kaggle.com/c/fake-news-pair-classification-challenge/data>)

## Weights

​	*<u>weights.npy</u>* is the weights for QuoraQP used in our paper. Weights for Train/Dev/Test sets are concatenated together. We recommend to use the QuoraQP released in [QuoraQP](<https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view>), since we notice that there are several versions of QuoraQP which are not exactly the same.

## Citation

If you use the code, please cite following paper,

```latex
@article{zhang2019selection,
  title={Selection Bias Explorations and Debias Methods for Natural Language Sentence Matching Datasets},
  author={Zhang, Guanhua and Bai, Bing and Liang, Jian and Bai, Kun and Chang, Shiyu and Yu, Mo and Zhu, Conghui and Zhao, Tiejun},
  journal={arXiv preprint arXiv:1905.06221},
  year={2019}
}
```