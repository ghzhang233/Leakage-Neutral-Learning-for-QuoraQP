import os
import pickle
import random
import re

import networkx as nx
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, Add
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from nltk.tag import pos_tag
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, scale

random.seed(233)
np.random.seed(233)
tf.set_random_seed(233)


def encode_data(data):
    data['label'] = LabelEncoder().fit_transform(data['label'])
    le = LabelEncoder()
    le.fit(data['sen1'].append(data['sen2']))
    data['tid1'] = le.transform(data['sen1'])
    data['tid2'] = le.transform(data['sen2'])
    return data


def extract_leakage(data):
    tid1 = data['tid1'].values
    tid2 = data['tid2'].values

    doc_number = np.max((tid1.max(), tid2.max())) + 1
    adj = coo_matrix((np.ones(len(tid1) * 2), (np.concatenate(
        [tid1, tid2]), np.concatenate([tid2, tid1]))), (doc_number, doc_number))

    degree = adj.sum(axis=0)
    data['tid1_degree'] = data['tid1'].apply(lambda x: degree[0, x])
    data['tid2_degree'] = data['tid2'].apply(lambda x: degree[0, x])

    tmp = adj * adj
    data['path1'] = data.apply(
        lambda row: tmp[int(row['tid1']), int(row['tid2'])], axis=1)
    degree1 = tmp.sum(axis=0)
    data['tid1_degree1'] = data['tid1'].apply(lambda x: degree1[0, x])
    data['tid2_degree1'] = data['tid2'].apply(lambda x: degree1[0, x])

    tmp = adj * adj * adj
    data['path2'] = data.apply(
        lambda row: tmp[int(row['tid1']), int(row['tid2'])], axis=1)
    degree2 = tmp.sum(axis=0)
    data['tid1_degree2'] = data['tid1'].apply(lambda x: degree2[0, x])
    data['tid2_degree2'] = data['tid2'].apply(lambda x: degree2[0, x])
    return data


def extract_unlexicalized(data):
    print("Extracting Unlexicalized Features...")

    def text_cleaning(text):
        text_cleaned = re.sub('[^A-Za-z0-9]', ' ', text.lower()).split()
        text_cleaned = ['$NUL$'] if len(text_cleaned) == 0 else text_cleaned
        return text_cleaned

    def get_bleu_n_gram(x, y, n_gram):
        return nltk.translate.bleu_score.sentence_bleu(
            [x], y, weights=[1 / n_gram for _ in range(n_gram)])

    def get_num_overlap(x, y, pos=None):
        if pos is not None:
            x = [i[0] for i in pos_tag(x) if i[1] == pos]
            y = [i[0] for i in pos_tag(y) if i[1] == pos]
        cnt = 0
        for i in x:
            for j in y:
                if i == j:
                    cnt += 1
        return cnt, (cnt / (len(x) * len(y))) if len(x) * len(y) != 0 else 0

    sen1 = data['sen1'].apply(text_cleaning).values
    sen2 = data['sen2'].apply(text_cleaning).values

    num_pair = len(sen1)

    bleu1 = [get_bleu_n_gram(sen1[i], sen2[i], 1) for i in range(num_pair)]
    bleu2 = [get_bleu_n_gram(sen1[i], sen2[i], 2) for i in range(num_pair)]
    bleu3 = [get_bleu_n_gram(sen1[i], sen2[i], 3) for i in range(num_pair)]
    bleu4 = [get_bleu_n_gram(sen1[i], sen2[i], 4) for i in range(num_pair)]
    length_dist = [(len(sen1[i]) - len(sen2[i])) for i in range(num_pair)]
    num_overlap = [get_num_overlap(sen1[i], sen2[i])[0]
                   for i in range(num_pair)]
    rate_overlap = [num_overlap[i] /
                    (len(sen1[i]) * len(sen2[i])) for i in range(num_pair)]
    num_overlap_noun = [get_num_overlap(sen1[i], sen2[i], 'NOUN')[
                            0] for i in range(num_pair)]
    rate_overlap_noun = [get_num_overlap(sen1[i], sen2[i], 'NOUN')[
                             1] for i in range(num_pair)]
    num_overlap_verb = [get_num_overlap(sen1[i], sen2[i], 'VERB')[
                            0] for i in range(num_pair)]
    rate_overlap_verb = [get_num_overlap(sen1[i], sen2[i], 'VERB')[
                             1] for i in range(num_pair)]
    num_overlap_adj = [get_num_overlap(sen1[i], sen2[i], 'ADJ')[
                           0] for i in range(num_pair)]
    rate_overlap_adj = [get_num_overlap(sen1[i], sen2[i], 'ADJ')[
                            1] for i in range(num_pair)]
    num_overlap_adv = [get_num_overlap(sen1[i], sen2[i], 'ADV')[
                           0] for i in range(num_pair)]
    rate_overlap_adv = [get_num_overlap(sen1[i], sen2[i], 'ADV')[
                            1] for i in range(num_pair)]

    data['bleu1'] = bleu1
    data['bleu2'] = bleu2
    data['bleu3'] = bleu3
    data['bleu4'] = bleu4
    data['length_dist'] = length_dist
    data['num_overlap'] = num_overlap
    data['rate_overlap'] = rate_overlap
    data['num_overlap_noun'] = num_overlap_noun
    data['rate_overlap_noun'] = rate_overlap_noun
    data['num_overlap_verb'] = num_overlap_verb
    data['rate_overlap_verb'] = rate_overlap_verb
    data['num_overlap_adj'] = num_overlap_adj
    data['rate_overlap_adj'] = rate_overlap_adj
    data['num_overlap_adv'] = num_overlap_adv
    data['rate_overlap_adv'] = rate_overlap_adv

    return data


def extract_deepwalk(data, name_dataset):
    print("Extracting Deepwalk Features...")
    tid1 = data['tid1'].values
    tid2 = data['tid2'].values
    num_nodes = np.max((tid1.max(), tid2.max())) + 1

    nbs = [[] for _ in range(num_nodes)]
    for u, v in zip(np.concatenate([tid1, tid2]), np.concatenate([tid2, tid1])):
        nbs[u].append(v)
    filename_adj = "./data/adj_%s.txt" % name_dataset
    filename_emb = "./data/deepwalk_emb_%s.txt" % name_dataset
    with open(filename_adj, "w", encoding="utf-8") as fout:
        for u in range(num_nodes):
            fout.write(str(u) + " ")
            fout.write(" ".join([str(v) for v in nbs[u]]) + "\n")
    print("deepwalk --input %s --output %s" % (filename_adj, filename_emb))
    os.system("deepwalk --input %s --output %s" % (filename_adj, filename_emb))

    dim_emb = 64
    emb_ret = np.zeros([num_nodes, dim_emb])
    with open(filename_emb, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 0:
                continue
            s = line.strip().split()
            u = eval(s[0])
            emb_ret[u] = np.array([eval(i) for i in s[1:]])

    data['emb1'] = [emb_ret[i] for i in tid1]
    data['emb2'] = [emb_ret[i] for i in tid2]
    emb_dot = np.sum(emb_ret[tid1] * emb_ret[tid2], axis=1)
    data['emb_dot'] = [emb_dot[i] for i in range(len(tid1))]
    return data


def extract_network_based(data):
    print("Extracting Network Based Fearure...")
    tid1 = data['tid1'].values
    tid2 = data['tid2'].values
    num_nodes = np.max((tid1.max(), tid2.max())) + 1

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for u, v in zip(np.concatenate([tid1, tid2]), np.concatenate([tid2, tid1])):
        G.add_edge(u, v)

    preds = nx.resource_allocation_index(G, list(zip(tid1, tid2)))
    data['nx1'] = [p for (u, v, p) in preds]
    preds = nx.jaccard_coefficient(G, list(zip(tid1, tid2)))
    data['nx2'] = [p for (u, v, p) in preds]
    preds = nx.preferential_attachment(G, list(zip(tid1, tid2)))
    data['nx3'] = [p for (u, v, p) in preds]

    G.add_node(num_nodes)
    for i in range(num_nodes):
        G.add_edge(i, num_nodes)
        G.add_edge(num_nodes, i)
    preds = nx.adamic_adar_index(G, list(zip(tid1, tid2)))
    data['nx4'] = [p for (u, v, p) in preds]
    return data


def read_data(name_dataset, use_saved=False):
    print("Reading %s..." % name_dataset)

    if use_saved:
        data = pickle.load(open("data/data_%s" % name_dataset, "rb"))
        split = pickle.load(open("data/split_%s" % name_dataset, "rb"))
        return data, split

    tr, val, tst = None, None, None
    if name_dataset == 'bytedance':
        data = pd.read_csv('./bytedance/train.csv',
                           usecols=['title1_en', 'title2_en', 'label'])
        data.columns = ['sen1', 'sen2', 'label']

    elif name_dataset == 'multinli_match':
        tr = pd.read_csv('./multinli/multinli_1.0_train.txt', delimiter='\t',
                         usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        val = pd.read_csv('./multinli/multinli_1.0_dev_matched.txt', delimiter='\t',
                          usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        tst = pd.read_csv('./multinli/multinli_0.9_test_matched_unlabeled.txt', delimiter='\t',
                          usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        data = pd.concat([tr, val, tst]).fillna('_')
        data.columns = ['label', 'sen1', 'sen2', 'pairID']

    elif name_dataset == 'multinli_mismatch':
        tr = pd.read_csv('./multinli/multinli_1.0_train.txt', delimiter='\t',
                         usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        val = pd.read_csv('./multinli/multinli_1.0_dev_mismatched.txt', delimiter='\t',
                          usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        tst = pd.read_csv('./multinli/multinli_0.9_test_mismatched_unlabeled.txt', delimiter='\t',
                          usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        data = pd.concat([tr, val, tst]).fillna('_')
        data.columns = ['label', 'sen1', 'sen2', 'pairID']

    elif name_dataset == 'quora':
        tr = pd.read_csv('./quora/train.tsv', delimiter='\t', header=None)
        val = pd.read_csv('./quora/dev.tsv', delimiter='\t', header=None)
        tst = pd.read_csv('./quora/test.tsv', delimiter='\t', header=None)
        data = pd.concat([tr, val, tst]).fillna('')
        data.columns = ['label', 'sen1', 'sen2', 'pair_id']

    elif name_dataset == 'sick':
        tr = pd.read_csv('./sick/SICK_train.txt', delimiter='\t',
                         usecols=['entailment_judgment', 'sentence_A', 'sentence_B'])
        val = pd.read_csv('./sick/SICK_trial.txt', delimiter='\t',
                          usecols=['entailment_judgment', 'sentence_A', 'sentence_B'])
        tst = pd.read_csv('./sick/SICK_test.txt', delimiter='\t',
                          usecols=['entailment_judgment', 'sentence_A', 'sentence_B'])
        data = pd.concat([tr, val, tst]).fillna('')
        data.columns = ['sen1', 'sen2', 'label']

    elif name_dataset == 'sick_sts':
        tr = pd.read_csv('./sick/SICK_train.txt', delimiter='\t',
                         usecols=['sentence_A', 'sentence_B', 'relatedness_score'])
        val = pd.read_csv('./sick/SICK_trial.txt', delimiter='\t',
                          usecols=['relatedness_score', 'sentence_A', 'sentence_B'])
        tst = pd.read_csv('./sick/SICK_test.txt', delimiter='\t',
                          usecols=['relatedness_score', 'sentence_A', 'sentence_B'])
        data = pd.concat([tr, val, tst]).fillna('')
        data.columns = ['sen1', 'sen2', 'label']
        data['label'] = data['label'].apply(lambda x: 1 if x > 3.6 else 0)

    elif name_dataset == 'snli':
        tr = pd.read_csv('./snli/snli_1.0_train.txt', delimiter='\t',
                         usecols=['gold_label', 'sentence1', 'sentence2'])
        val = pd.read_csv('./snli/snli_1.0_dev.txt', delimiter='\t',
                          usecols=['gold_label', 'sentence1', 'sentence2'])
        tst = pd.read_csv('./snli/snli_1.0_test.txt', delimiter='\t',
                          usecols=['gold_label', 'sentence1', 'sentence2'])
        data = pd.concat([tr, val, tst]).fillna('_')
        data.columns = ['label', 'sen1', 'sen2']

    elif name_dataset == 'msr':
        tr = pd.read_csv('./msr/msr_paraphrase_train.txt', delimiter='\t',
                         usecols=['#1 String', '#2 String', 'Quality'])
        tst = pd.read_csv('./msr/msr_paraphrase_test.txt', delimiter='\t',
                          usecols=['#1 String', '#2 String', 'Quality'])
        data = pd.concat([tr, tst]).fillna('_')
        data.columns = ['label', 'sen1', 'sen2']

    else:
        print("Wrong Name!")
        return

    data = encode_data(data)
    data = extract_leakage(data)
    data = extract_unlexicalized(data)
    data = extract_network_based(data)
    data = extract_deepwalk(data, name_dataset)

    if name_dataset == 'bytedance':
        num_samples = len(data)
        perm = np.random.permutation(num_samples)
        train_split = perm[:int(0.8 * num_samples)]
        val_split = perm[int(0.8 * num_samples):int(0.9 * num_samples)]
        test_split = perm[int(0.9 * num_samples):]
    elif name_dataset == 'msr':
        num_samples_tr = len(tr)
        num_samples_ts = len(tst)
        perm = np.random.permutation(num_samples_tr)
        train_split = perm[:-1000]
        val_split = perm[-1000:]
        test_split = np.arange(num_samples_tr, num_samples_tr + num_samples_ts)
    else:
        num_samples_tr = len(tr)
        num_samples_vl = len(val)
        num_samples_ts = len(tst)
        train_split = np.arange(num_samples_tr)
        val_split = np.arange(num_samples_tr, num_samples_tr + num_samples_vl)
        test_split = np.arange(num_samples_tr + num_samples_vl,
                               num_samples_tr + num_samples_vl + num_samples_ts)
    split = (train_split, val_split, test_split)

    pickle.dump(data, open("data/data_%s" % name_dataset, "wb"))
    pickle.dump(split, open("data/split_%s" % name_dataset, "wb"))

    return data, split


def get_model(dim_in, dim_out):
    model_in = Input(shape=(dim_in,), dtype='float32')

    hidden = model_in
    for i in range(5):
        hidden_in = hidden
        hidden = Dense(128, activation='relu', use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        if i != 0:
            hidden = Add()([hidden, hidden_in])

    hidden = Dropout(0.5)(hidden)
    preds = Dense(dim_out, activation='softmax')(hidden)

    ret_model = Model(inputs=[model_in], outputs=preds)
    ret_model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=1e-3), metrics=['acc'])
    return ret_model


def train_and_evaluate(name_dataset, data, split, feature_mode="leakage"):
    print("Training...")
    if feature_mode == "leakage":
        X = data[['tid1_degree', 'tid2_degree', 'path1']].values
    elif feature_mode == "leakage1":
        X = data[['tid1_degree']].values
    elif feature_mode == "leakage2":
        X = data[['tid2_degree']].values
    elif feature_mode == "leakage3":
        X = data[['path1']].values
    elif feature_mode == "leakage12":
        X = data[['tid1_degree', 'tid2_degree']].values
    elif feature_mode == "leakage13":
        X = data[['tid1_degree', 'path1']].values
    elif feature_mode == "leakage23":
        X = data[['tid2_degree', 'path1']].values
    elif feature_mode == "unlexicalized":
        X = data[['bleu1', 'bleu2', 'bleu3', 'bleu4', 'length_dist',
                  'num_overlap', 'rate_overlap', 'num_overlap_noun',
                  'rate_overlap_noun', 'num_overlap_verb', 'rate_overlap_verb',
                  'num_overlap_adj', 'rate_overlap_adj', 'num_overlap_adv', 'rate_overlap_adv',
                  ]].values
    elif feature_mode == "link_prediction":
        X = data[['tid1_degree', 'tid2_degree', 'path1', 'tid1_degree1', 'tid2_degree1',
                  'path2', 'tid1_degree2', 'tid2_degree2', 'nx1', 'nx2', 'nx3', 'nx4', 'emb_dot']].values
        emb1 = np.array([data['emb1'].values[i]
                         for i in range(len(data['emb1'].values))])
        emb2 = np.array([data['emb2'].values[i]
                         for i in range(len(data['emb2'].values))])
        X = np.concatenate([X, emb1 * emb2], axis=1)
    else:
        print("Feature Error!")
        return
    X = scale(X)
    y = data['label'].values
    train_split, val_split, test_split = split

    use_mlp = False
    if use_mlp:
        model = get_model(X.shape[1], y.max() + 1)
        best_model_dir = './data/model_%s_%s.h5' % (name_dataset, feature_mode)
        early_stopping = EarlyStopping(
            monitor='val_acc', mode='max', patience=10)
        model_checkpoint = ModelCheckpoint(best_model_dir, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_acc', mode='max', factor=0.2, patience=1, min_lr=1e-5, verbose=0)
        model.fit(X[train_split], to_categorical(y[train_split]),
                  validation_data=(X[val_split], to_categorical(y[val_split])), verbose=0,
                  epochs=100, batch_size=256, shuffle=True, callbacks=[early_stopping, model_checkpoint, reduce_lr])
        model.load_weights(best_model_dir)

        tr_acc = accuracy_score(y[train_split],
                                np.argmax(model.predict(X[train_split]), axis=1))
        val_acc = accuracy_score(y[val_split],
                                 np.argmax(model.predict(X[val_split]), axis=1))
        y_pred = np.argmax(model.predict(X[test_split]), axis=1)
    else:
        if name_dataset in {'sick', 'sick_sts', 'msr'}:
            if feature_mode == 'link_prediction':
                model = RandomForestClassifier(
                    n_estimators=500, max_depth=10, random_state=0, n_jobs=-1)
            else:
                model = RandomForestClassifier(
                    n_estimators=500, max_depth=5, random_state=0, n_jobs=-1)
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=30, random_state=0, n_jobs=-1)
        model.fit(X[np.concatenate([train_split, val_split])],
                  y[np.concatenate([train_split, val_split])])
        tr_acc = accuracy_score(y[train_split], model.predict(X[train_split]))
        val_acc = accuracy_score(y[val_split], model.predict(X[val_split]))
        y_pred = model.predict(X[test_split])

    print('Best Train Acc: %.5lf' % tr_acc)
    print('Best Valid Acc: %.5lf' % val_acc)

    if name_dataset in {'multinli_match', 'multinli_mismatch'}:
        with open("multinli_results/y_pred_%s_%s.txt" % (name_dataset, feature_mode), "w", encoding='utf-8') as fout:
            for i in range(len(y_pred)):
                fout.write(
                    str(data['pairID'].values[test_split[i]]) + "," + str(y_pred[i]) + "\n")
        test_acc = 0
        test_guess_acc = 0
    else:
        test_acc = accuracy_score(y[test_split], y_pred)
        # freq_label = np.argmax([np.mean(i == y[np.concatenate([train_split, val_split])]) for i in set(y)])
        freq_label = np.argmax([np.mean(i == y[test_split]) for i in set(y)])
        test_guess_acc = np.mean(freq_label == y[test_split])
        print('Best Test Acc: %.5lf' % test_acc)
        print('Best Guess of Test Acc: %.5lf' % test_guess_acc)

    if name_dataset == 'quora' and feature_mode == 'leakage':
        sample_weight = np.load('./weights.npy')
        print("Sample Weight Max: %.5lf Min: %.5lf" %
              (np.max(sample_weight), np.min(sample_weight)))
        test_weighted_acc = accuracy_score(
            y[test_split], y_pred, sample_weight=sample_weight[test_split])
        # freq_label = np.argmax([np.mean(i == y[np.concatenate([train_split, val_split])]) for i in set(y)])
        freq_label = np.argmax([np.mean(i == y[test_split]) for i in set(y)])
        y_guess_pred = np.ones(len(y[test_split])) * freq_label
        test_guess_weighted_acc = accuracy_score(y[test_split], y_guess_pred, sample_weight=sample_weight[test_split])
        print("Best Test Weighted Acc: %.5lf" % test_weighted_acc)
        print("Best Guess of Test Weighted Acc: %.5lf" %
              test_guess_weighted_acc)

    return tr_acc, val_acc, test_acc, test_guess_acc


def run(ret, name_dataset, use_saved=False):
    data, split = read_data(name_dataset, use_saved)
    if name_dataset in {'snli', 'quora', 'sick', 'sick_sts', 'bytedance'}:
        feature_mode_list = ["leakage", "unlexicalized", "link_prediction", "leakage1",
                             "leakage2", "leakage3", "leakage12", "leakage13", "leakage23"]
    else:
        feature_mode_list = ["leakage", "unlexicalized", "link_prediction"]
    for feature_mode in feature_mode_list:
        tr_acc, val_acc, test_acc, test_guess_acc = train_and_evaluate(
            name_dataset, data, split, feature_mode)
        ret.loc["%s_%s" % (name_dataset, feature_mode)] = [
            tr_acc, val_acc, test_acc, test_guess_acc]
        ret.to_csv("result.csv")

    return ret


if __name__ == "__main__":
    ret = pd.DataFrame(
        columns=['tr_acc', 'val_acc', 'test_acc', 'test_guess_acc'])
    ret = run(ret, 'snli')
    ret = run(ret, 'multinli_match')
    ret = run(ret, 'multinli_mismatch')
    ret = run(ret, 'quora')
    ret = run(ret, 'msr')
    ret = run(ret, 'sick')
    ret = run(ret, 'sick_sts')
    ret = run(ret, 'bytedance')
    print(ret[['test_acc', 'test_guess_acc']])
    ret.to_csv("result.csv")
