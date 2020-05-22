import io
import os
import re
import keras
import random

import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from sklearn.preprocessing import scale

########################################
# parameters
########################################
random.seed(1234)
np.random.seed(1234)

split_ratio = -1

max_nb_words = 50000
max_seq_len = 35
emb_dim = 300

dir_base = './data/'
file_emb = dir_base + 'wordvec.txt'
file_train = dir_base + 'train.tsv'
file_val = dir_base + 'dev.tsv'
file_test = dir_base + 'test.tsv'
file_sick = dir_base + 'sick.txt'
file_msr = dir_base + 'msr.txt'
file_sample_weight = dir_base + 'density.npy'

dir_processed = "./processed_data/"
if not os.path.isdir(dir_processed):
    os.mkdir(dir_processed)

stamp_data = str(split_ratio)

file_data = dir_processed + 'data_%s.npz' % str(split_ratio)
file_split = dir_processed + 'split_%s.npz' % str(split_ratio)
file_leaky = dir_processed + 'leakage_features_%s.npz' % str(split_ratio)

########################################
# read-data
########################################

tr = pd.read_csv(file_train, delimiter='\t', header=None)
tr.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
tr = tr[['is_duplicate', 'question1', 'question2']]

val = pd.read_csv(file_val, delimiter='\t', header=None)
val.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
val = val[['is_duplicate', 'question1', 'question2']]

tst = pd.read_csv(file_test, delimiter='\t', header=None)
tst.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
tst = tst[['is_duplicate', 'question1', 'question2']]

sick = pd.read_csv(file_sick, delimiter='\t', usecols=['sentence_A', 'sentence_B', 'relatedness_score'])
sick.columns = ['question1', 'question2', 'is_duplicate']
sick['is_duplicate'] = sick['is_duplicate'].apply(lambda x: 1 if x > 3.6 else 0)

msr = pd.read_csv(file_msr, delimiter='\t', usecols=['#1 String', '#2 String', 'Quality'])
msr.columns = ['is_duplicate', 'question1', 'question2']
data = pd.concat([tr, val, tst, sick, msr], sort=False).fillna('')

########################################
# pre-processing
########################################

print('Pre-processing')


def text_cleaning(text):
    text = re.sub('[^A-Za-z0-9]', ' ', text.lower())
    text = ' '.join(text.split())
    return text


data['question1'] = data['question1'].apply(text_cleaning)
data['question2'] = data['question2'].apply(text_cleaning)
tokenizer = Tokenizer(num_words=max_nb_words, oov_token='oov_token_placeholder')
tokenizer.fit_on_texts(list(data['question1'].values) + list(data['question2'].values))
sequences_1 = tokenizer.texts_to_sequences(data['question1'].values)
sequences_2 = tokenizer.texts_to_sequences(data['question2'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
x1 = pad_sequences(sequences_1, maxlen=max_seq_len)
x2 = pad_sequences(sequences_2, maxlen=max_seq_len)
y = data['is_duplicate'].values

########################################
# retrieval embeddings
########################################

print('Indexing word vectors')
word2vec = {}
fin = io.open(file_emb, 'r', encoding='utf-8', newline='\n', errors='ignore')
for line in fin:
    tokens = line.rstrip().split(' ')
    word2vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
print('Found %s word vectors of word2vec' % len(word2vec.keys()))
print('Preparing embedding matrix')
nb_words = min(max_nb_words, len(word_index))

emb = np.zeros((nb_words + 1, emb_dim))
miss_cnt = 0
for word, i in word_index.items():
    if i >= nb_words:
        break
    if word in word2vec.keys():
        emb[i] = word2vec[word]
    else:
        emb[i] = (np.random.rand(emb_dim) - 0.5) * 0.1
        miss_cnt += 1
print('Null word embeddings: %d' % miss_cnt)

########################################
# sample train/val/test data
########################################

questions = list(data['question1'].values) + list(data['question2'].values)
le = preprocessing.LabelEncoder()
le.fit(questions)
q1_id = le.transform(data['question1'].values)
q2_id = le.transform(data['question2'].values)
pair_number = q1_id.shape[0]
sen_number = np.max((q1_id.max(), q2_id.max())) + 1

num_data = len(tr) + len(val) + len(tst)
sick_idx = np.arange(num_data, num_data + len(sick))
msr_idx = np.arange(num_data + len(sick), num_data + len(sick) + len(msr))

if split_ratio == -1:
    train_idx = np.arange(len(tr))
    val_idx = np.arange(len(tr), len(tr) + len(val))
    test_idx = np.arange(len(tr) + len(val), len(tr) + len(val) + len(tst))
else:
    perm = np.random.permutation(num_data)
    val_split = (1 - split_ratio) / 2
    train_idx = perm[:int(num_data * split_ratio)]
    val_idx = perm[int(num_data * split_ratio): int(num_data * (split_ratio + val_split))]
    test_idx = perm[int(num_data * (split_ratio + val_split)):]

train_sent_set = set(q1_id[train_idx]) | set(q2_id[train_idx])
val_overlap_idx = [i for i, idx in enumerate(val_idx) if
                   (q1_id[idx] in train_sent_set or q2_id[idx] in train_sent_set)]
test_overlap_idx = [i for i, idx in enumerate(test_idx) if
                    (q1_id[idx] in train_sent_set or q2_id[idx] in train_sent_set)]
val_no_overlap_idx = [i for i, idx in enumerate(val_idx) if
                      not (q1_id[idx] in train_sent_set or q2_id[idx] in train_sent_set)]
test_no_overlap_idx = [i for i, idx in enumerate(test_idx) if
                       not (q1_id[idx] in train_sent_set or q2_id[idx] in train_sent_set)]
print("Valid Overlap Distribution: %.5lf%%"
      % (y[val_idx][val_overlap_idx].sum() / len(val_overlap_idx) * 100.0))
print("Test Overlap Distribution: %.5lf%%" %
      (y[test_idx][test_overlap_idx].sum() / len(test_overlap_idx) * 100.0))
print("Valid No Overlap Distribution: %.5lf%%" %
      (y[val_idx][val_no_overlap_idx].sum() / len(val_no_overlap_idx) * 100.0))
print("Test No Overlap Distribution: %.5lf%%" %
      (y[test_idx][test_no_overlap_idx].sum() / len(test_no_overlap_idx) * 100.0))

sent_test_same = list(
    set(list(data['question1'].values[train_idx]) + list(data['question2'].values[train_idx])))
sequences_test_same = tokenizer.texts_to_sequences(sent_test_same)
x_test_same = pad_sequences(sequences_test_same, maxlen=max_seq_len)
y_test_same = np.ones(len(x_test_same))
test_same_idx = range(len(x1), len(x1) + len(x_test_same))
x1 = np.concatenate([x1, x_test_same])
x2 = np.concatenate([x2, x_test_same])
y = np.concatenate([y, y_test_same])

########################################
# process leaky feature
########################################

adj = coo_matrix((np.ones(len(q1_id) * 2), (np.concatenate(
    [q1_id, q2_id]), np.concatenate([q2_id, q1_id]))), (sen_number, sen_number))

leaky_features = np.zeros([len(q1_id), 8])
degree = np.array(adj.sum(axis=1))
leaky_features[:, 0] = degree[q1_id][:, 0]
leaky_features[:, 1] = degree[q2_id][:, 0]

tmp = adj * adj
degree1 = np.array(tmp.sum(axis=1))
leaky_features[:, 2] = np.array([tmp[q1_id[i], q2_id[i]] for i in range(len(q1_id))])
leaky_features[:, 3] = degree1[q1_id][:, 0]
leaky_features[:, 4] = degree1[q2_id][:, 0]

tmp = adj * adj * adj
degree2 = np.array(tmp.sum(axis=1))
leaky_features[:, 5] = np.array([tmp[q1_id[i], q2_id[i]] for i in range(len(q1_id))])
leaky_features[:, 6] = degree1[q1_id][:, 0]
leaky_features[:, 7] = degree1[q2_id][:, 0]
leaky_features = leaky_features[:, :3]
leaky_features = scale(leaky_features)

########################################
# save data to disk
########################################

np.savez(file_data, x1=x1, x2=x2, y=y, emb=emb, word_index=word_index)
np.savez(file_split, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
         val_overlap_idx=val_overlap_idx, val_no_overlap_idx=val_no_overlap_idx,
         test_overlap_idx=test_overlap_idx, test_no_overlap_idx=test_no_overlap_idx,
         sick_idx=sick_idx, msr_idx=msr_idx, test_same_idx=test_same_idx)
np.savez(file_leaky, leaky_features=leaky_features)
