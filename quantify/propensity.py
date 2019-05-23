import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.sparse import coo_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import log_loss, make_scorer
from sklearn.ensemble import RandomForestClassifier
from math import fabs

tr = pd.read_csv('./quora/train.tsv', delimiter='\t', header=None)
tr.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
val = pd.read_csv('./quora/dev.tsv', delimiter='\t', header=None)
val.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
te = pd.read_csv('./quora/test.tsv', delimiter='\t', header=None)
te.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
data = pd.concat([tr, val, te]).fillna('')

questions = list(data['question1'].values) + list(data['question2'].values)
le = preprocessing.LabelEncoder()
le.fit(questions)
data['q1_id'] = le.transform(data['question1'].values)
data['q2_id'] = le.transform(data['question2'].values)


def leaky_extracting(concat):

    tid1 = concat['q1_id'].values
    tid2 = concat['q2_id'].values

    doc_number = np.max((tid1.max(), tid2.max())) + 1
    adj = coo_matrix((np.ones(len(tid1) * 2), (np.concatenate(
        [tid1, tid2]), np.concatenate([tid2, tid1]))), (doc_number,  doc_number))

    degree = adj.sum(axis=0)
    concat['q1_id_dgree'] = concat['q1_id'].apply(lambda x: degree[0, x])
    concat['q2_id_dgree'] = concat['q2_id'].apply(lambda x: degree[0, x])

    tmp = adj * adj
    concat['path'] = concat.apply(
        lambda row: tmp[int(row['q1_id']), int(row['q2_id'])], axis=1)

    return concat


data = leaky_extracting(data)

X = data[['q1_id_dgree', 'q2_id_dgree', 'path']].values
XX = np.zeros_like(X)
XX[:, 0] = X[:, 0] * X[:, 1]
XX[:, 1] = X[:, 0] * X[:, 2]
XX[:, 2] = X[:, 1] * X[:, 2]
X = np.hstack((X, XX))
y = data['is_duplicate'].values

clf = RandomForestClassifier(
    n_estimators=10000, max_depth=9, random_state=2018, n_jobs=8, criterion='entropy')
time_start=time.time()
y_pred = cross_val_predict(clf, X, y, cv=100, method='predict_proba', verbose=3, n_jobs=1)
time_end = time.time()
propensity = np.array([y_pred[i, y[i]] for i in range(len(y))])
print('totally cost', time_end-time_start)
print(np.mean(np.log(propensity)))
np.save('propensity.npy', propensity)

# propensity = np.load("propensity.npy")
prob_1_l = np.array([(propensity[i] if y[i] == 1 else (1-propensity[i]))
          for i in range(len(y))])
prob_0_l = 1 - prob_1_l


def calculate_weight_fraction(prob_1):
    prob_0 = 1 - prob_1
    w1 = 1 / (prob_0 * prob_1_l / (prob_0 * prob_1_l + prob_1 * prob_0_l))
    w0 = 1 / (prob_1 * prob_0_l / (prob_0 * prob_1_l + prob_1 * prob_0_l))
    return sum(w1[i] for i in range(len(y)) if y[i] == 1) / sum(w0[i] for i in range(len(y)) if y[i] == 0)


prior_fraction = np.sum(y) / (len(y) - np.sum(y))
l, r = 0, 1
thr = 0.00000000001
step = 100
# while l + thr < r:
for _ in range(step):
    m1 = l + (r- l) / 2
    if calculate_weight_fraction(m1) < prior_fraction:
        l = m1
    else:
        r = m1

m0 = 1 - m1
w1 = 1 / (m0 * prob_1_l / (m0 * prob_1_l + m1 * prob_0_l))
w0 = 1 / (m1 * prob_0_l / (m0 * prob_1_l + m1 * prob_0_l))
weights = np.array([(w1[i] if y[i] == 1 else w0[i]) for i in range(len(y))])
assert fabs(prior_fraction - sum([weights[i] for i in range(len(y)) if y[i] == 1]) / sum([weights[i] for i in range(len(y)) if y[i] == 0])) < 0.0001
np.save(open("weights.npy", "wb"), weights)

