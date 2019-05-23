# Quantifying the Biasedness in Datasets

Codes for <u>generating weights</u> and <u>predicting with leakage in six NLSM datasest</u>.

## Environment

```
	tensorflow>=1.12.1
	networkx==2.1
	Keras==2.2.4
	scipy==1.1.0
	nltk==3.3
	numpy==1.14.3
	pandas==0.23.0
	scikit_learn==0.21.1
```

If you want to use deepwalk features, you also need to install [deepwalk](<https://github.com/phanein/deepwalk>) and add it to PATH. It is worth mentioning that calculating deepwalk embeddings is very slow. If you don not want to use it, you can simply comment the line 287 out and remove *'emb_dot'* in line 362.

## Data

Download the datasets and manage the files like below:

```
./
│ leaky_predict.py
│ weights.npy
| requirements.txt
├─bytedance
│      train.csv
├─msr
│      msr_paraphrase_test.txt
│      msr_paraphrase_train.txt
├─multinli
│      multinli_0.9_test_matched_unlabeled.txt
│      multinli_0.9_test_mismatched_unlabeled.txt
│      multinli_1.0_dev_matched.txt
│      multinli_1.0_dev_mismatched.txt
│      multinli_1.0_train.txt
├─quora
│      dev.tsv
│      test.tsv
│      train.tsv
├─sick
│      SICK_test.txt
│      SICK_train.txt
│      SICK_trial.txt
└─snli
       snli_1.0_dev.txt
       snli_1.0_test.txt
       snli_1.0_train.txt
```

## Usage

```
# for generating weights
python propensity.py
# for predict with leakage in six datasets
python leaky_predict.py
```

