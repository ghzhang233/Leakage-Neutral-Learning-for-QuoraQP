# Leakage-neutral Learning Method

Codes for <u>leakage-neutral learning in QuoraQP</u>.

## Environment

```
	numpy==1.14.3
	pandas==0.23.0
	scipy==1.1.0
	Keras==2.2.4
	tensorflow==1.11.0
	scikit_learn==0.21.1
```

## Data

Download the datasets and manage the files like below:

```
./
│ main.py
│ make_data.py
│ utils.py
│ requirements.txt
├─data
│      dev.tsv
│      msr.txt
│      sick.txt
│      test.tsv
│      train.tsv
│      wordvec.txt
├─processed_data
       weights.npy
```

<u>**trian/test/dev.tsv**</u>  and **<u>wordvec.txt</u>** can be found in [QuoraQP.](<https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view>)

<u>**msr.txt**</u> are made by concatenating <u>msr_paraphrase_test.txt</u> and <u>msr_paraphrase_train.txt</u> from [MSRP](<https://www.microsoft.com/en-us/download/details.aspx?id=52398>).

<u>**sick.txt**</u> are made by concatenating <u>SICK_train.txt</u>, <u>SICK_trail.txt</u> and <u>SICK_test.txt</u> from [SICK](<http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools>).

## Usage

```
PYTHONHASHSEED=0 python make_data.py

# for baseline
PYTHONHASHSEED=0 python main.py

# for leakage-neutral learning
PYTHONHASHSEED=0 python main.py --use_sample_weight_train --data_gen_mode prob
```

If you run without GPU, add following argument:

```
--not_use_cndnn
```

If you want to specify random seeds, use following argument:

```
--random_seed seed_number	# default 0
```

We also tried using adversarial learning to mitigate the bias pattern. However, it did not work. We owe the failure to that the leakage features are strongly correlated with labels, which means not only leakage feature can be used to predict labels, but also labels can be used to predict the leakage features in the origin biased distribution. So when models cannot predict the leakage features, they also miss the labels, leading to a significant drop on performance. We did not obtain the experimental results of adversarial learning in the paper, but if you are interested, just add arguments as following,

```
PYTHONHASHSEED=0 python main.py --pred_weight 1.0 --adv_weight 1.0
```

