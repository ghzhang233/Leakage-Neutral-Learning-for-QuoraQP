import argparse
import os
import random
import time

import keras
import keras.backend.tensorflow_backend as ktf
import numpy as np
import tensorflow as tf
from keras.layers import Dropout
from keras.layers import Input, Dense, Embedding, CuDNNLSTM, LSTM
from keras.layers import concatenate, dot, BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from utils import ResultRecorder, DataGenerator, get_logger

########################################
# add argparse
########################################

parser = argparse.ArgumentParser()
parser.add_argument('--use_loaded_model', action='store_true', help='load model')
parser.add_argument('--model_stamp', type=str, default="NULL", help='set stamp for load model')
parser.add_argument('--not_use_cudnn', action='store_false', help='use cudnn lstm')
parser.add_argument('--random_seed', type=int, default=0, help='random seed.')
parser.add_argument('--split_ratio', type=float, default=-1, help='split ratio')
parser.add_argument('--adv_mode', type=str, default="mean", help='how to adv', choices=['inv', 'mean'])

# parameter about train
parser.add_argument('--valid_mode', type=str, default="all", help='which valid set',
                    choices=['all', 'no_overlap', 'overlap'])
parser.add_argument('--valid_std', type=str, default="acc", help='valid loss/acc/auc',
                    choices=['lss', 'acc', 'auc'])
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--max_no_improve', type=int, default=20, help='max epoch that do not improve')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.2, help='decay rate of learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
parser.add_argument('--pred_weight', type=float, default=0, help='coef of pred loss')
parser.add_argument('--adv_weight', type=float, default=0, help='coef of adv loss')
parser.add_argument('--clipping', type=float, default=5.0, help='clipping norm')

# parameter about model
parser.add_argument('--dim_hidden', type=int, default=128, help='dimension of hidden layer')
parser.add_argument('--num_lstm', type=int, default=1, help='number of lstm layers')
parser.add_argument('--num_f_hidden', type=int, default=1, help='number of hidden layers in hidden net')
parser.add_argument('--num_g_hidden', type=int, default=0, help='number of hidden layers in label out net')
parser.add_argument('--num_k_hidden', type=int, default=1, help='number of hidden layers in leaky pred net')

# parameter about sample weight
parser.add_argument('--use_sample_weight_train', action='store_true', help='use sample weight during train')
parser.add_argument('--data_gen_mode', type=str, default='normal', help='data generate mode',
                    choices=['normal', 'prob'])
args = parser.parse_args()

########################################
# parameters
########################################

model_stamp = args.model_stamp
use_loaded_model = args.use_loaded_model
use_sample_weight_train = args.use_sample_weight_train
use_cudnn = args.not_use_cudnn
random_seed = args.random_seed
valid_mode = args.valid_mode
valid_std = args.valid_std
split_ratio = args.split_ratio
adv_mode = args.adv_mode
batch_size = args.batch_size
lr = args.lr
lr_decay = args.lr_decay
max_no_improve = args.max_no_improve
dim_hidden = args.dim_hidden
num_lstm = args.num_lstm
num_f_hidden = args.num_f_hidden
num_g_hidden = args.num_g_hidden
num_k_hidden = args.num_k_hidden
dropout_rate = args.dropout_rate
pred_weight = args.pred_weight
adv_weight = args.adv_weight
num_epochs = args.num_epochs
clipping = args.clipping
data_gen_mode = args.data_gen_mode

max_nb_words = 50000
max_seq_len = 35
emb_dim = 300

dir_log = "./logs/"
dir_model = "./trained_models/"
dir_data = "./processed_data/"
if not os.path.isdir(dir_log):
    os.mkdir(dir_log)
if not os.path.isdir(dir_model):
    os.mkdir(dir_model)

stamp = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())) + "_" + str(random.randint(0, 1000))
stamp_model = model_stamp if use_loaded_model else stamp

file_log = dir_log + 'log_%s.txt' % stamp

file_data = dir_data + 'data_%s.npz' % str(split_ratio)
file_split = dir_data + 'split_%s.npz' % str(split_ratio)
file_leaky = dir_data + 'leakage_features_%s.npz' % str(split_ratio)
file_sample_weight = dir_data + 'weights.npy'

file_adn = dir_model + 'adn_%s.npz' % stamp_model
file_hidden_net = dir_model + 'hidden_net_%s.npz' % stamp_model
file_out_net = dir_model + 'out_net_%s.npz' % stamp_model
file_pred_net = dir_model + 'pred_net_%s.npz' % stamp_model
file_lp_net = dir_model + 'lp_net_%s.npz' % stamp_model

file_result = './results_summary.csv'

########################################
# report-parameter
########################################

logger = get_logger(file_log)
rec = ResultRecorder(logger, file_result)

rec.add("use_loaded_model", use_loaded_model)
rec.add("model_stamp", model_stamp)
rec.add("use_sample_weight_train", use_sample_weight_train)
rec.add("use_cudnn", use_cudnn)
rec.add("random_seed", random_seed)
rec.add("valid_mode", valid_mode)
rec.add("valid_std", valid_std)
rec.add("split_ratio", split_ratio)
rec.add("adv_mode", adv_mode)
rec.add("batch_size", batch_size)
rec.add("lr", lr)
rec.add("lr_decay", lr_decay)
rec.add("max_no_improve", max_no_improve)
rec.add("dim_hidden", dim_hidden)
rec.add("num_lstm", num_lstm)
rec.add("num_f_hidden", num_f_hidden)
rec.add("num_g_hidden", num_g_hidden)
rec.add("num_k_hidden", num_k_hidden)
rec.add("dropout_rate", dropout_rate)
rec.add("pred_weight", pred_weight)
rec.add("adv_weight", adv_weight)
rec.add("file_log", file_log)
rec.add("file_data", file_data)
rec.add("file_split", file_split)
rec.add("file_leaky", file_leaky)
rec.add("file_sample_weight", file_sample_weight)
rec.add("file_lp_net", file_lp_net)
rec.add("file_adn", file_adn)
rec.add("file_pred_net", file_pred_net)
rec.add("file_hidden_net", file_hidden_net)
rec.add("file_out_net", file_out_net)
rec.add("max_nb_words", max_nb_words)
rec.add("max_seq_len", max_seq_len)
rec.add("emb_dim", emb_dim)
rec.add("num_epochs", num_epochs)
rec.add("clipping", clipping)
rec.add("data_gen_mode", data_gen_mode)

########################################
# load data
########################################

loaded_data = np.load(file_data)
x1 = loaded_data["x1"]
x2 = loaded_data["x2"]
y = loaded_data["y"]
emb = loaded_data["emb"]
word_index = loaded_data["word_index"]
loaded_data = np.load(file_split)
train_idx = loaded_data["train_idx"]
val_idx = loaded_data["val_idx"]
test_idx = loaded_data["test_idx"]
val_overlap_idx = loaded_data["val_overlap_idx"]
val_no_overlap_idx = loaded_data["val_no_overlap_idx"]
test_overlap_idx = loaded_data["test_overlap_idx"]
test_no_overlap_idx = loaded_data["test_no_overlap_idx"]
test_same_idx = loaded_data["test_same_idx"]
sick_idx = loaded_data["sick_idx"]
msr_idx = loaded_data["msr_idx"]
loaded_data = np.load(file_leaky)
leaky_features = loaded_data["leaky_features"]

val_idx_origin = val_idx
if valid_mode == "overlap":
    val_idx = val_idx[val_overlap_idx]
if valid_mode == "no_overlap":
    val_idx = val_idx[val_no_overlap_idx]

if adv_mode == 'inv':
    leaky_features_adv = -leaky_features
else:
    leaky_features_adv = np.tile(np.mean(leaky_features, axis=0), (leaky_features.shape[0], 1))

sample_weight = np.load(file_sample_weight)

########################################
# set seed and gpu
########################################

np.random.seed(random_seed)
random.seed(random_seed)
tf.set_random_seed(random_seed)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.45
session = tf.Session(config=gpu_config)
ktf.set_session(session)


########################################
# define the model structure
########################################

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def make_model_f(embedding):
    input_1 = Input(shape=(max_seq_len,), dtype='int32')
    input_2 = Input(shape=(max_seq_len,), dtype='int32')

    embedding_layer = Embedding(embedding.shape[0],
                                embedding.shape[1],
                                weights=[embedding],
                                trainable=False,
                                input_length=max_seq_len)
    hidden_1 = embedding_layer(input_1)
    hidden_2 = embedding_layer(input_2)

    for j in range(num_lstm):
        lstm_cell = LSTM
        if use_cudnn:
            lstm_cell = CuDNNLSTM
        if j != num_lstm - 1:
            lstm_layer = lstm_cell(dim_hidden, return_sequences=True)
        else:
            lstm_layer = lstm_cell(dim_hidden, return_sequences=False)
        hidden_1 = lstm_layer(hidden_1)
        hidden_2 = lstm_layer(hidden_2)

    hidden = concatenate([hidden_1, hidden_2, dot([hidden_1, hidden_2], axes=1)])
    for _ in range(num_f_hidden):
        hidden = Dense(dim_hidden, activation='relu')(hidden)
        hidden = BatchNormalization()(hidden)

    ret_model = Model(inputs=[input_1, input_2], outputs=hidden)
    ret_model.compile(loss='mse', optimizer='rmsprop')

    logger.info("Model F:")
    ret_model.summary(print_fn=logger.info)

    return ret_model, hidden


def make_model_g():
    model_in = Input(shape=(dim_hidden,)) if num_f_hidden != 0 else Input(shape=(dim_hidden * 2 + 1,))
    hidden = model_in
    for _ in range(num_g_hidden):
        hidden = Dense(dim_hidden, activation='relu')(hidden)
        hidden = BatchNormalization()(hidden)
    hidden = Dropout(dropout_rate)(hidden)
    model_out = Dense(1, activation='sigmoid')(hidden)

    model = Model(model_in, model_out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    logger.info("Model G:")
    model.summary(print_fn=logger.info)

    return model, model_out


def make_model_k():
    model_in = Input(shape=(dim_hidden,)) if num_f_hidden != 0 else Input(shape=(dim_hidden * 2 + 1,))
    hidden = model_in
    for _ in range(num_k_hidden):
        hidden = Dense(dim_hidden, activation='relu')(hidden)
        hidden = BatchNormalization()(hidden)
    hidden = Dropout(dropout_rate)(hidden)
    model_out = Dense(3, activation=None)(hidden)

    model = Model(model_in, model_out)
    model.compile(loss='mse', loss_weights=[pred_weight], optimizer=RMSprop(lr=lr, clipnorm=clipping))

    logger.info("Model K:")
    model.summary(print_fn=logger.info)

    return model, model_out


def make_model_adn(model_f, model_g, model_k):
    model_input_1 = Input(shape=(max_seq_len,), dtype='int32')
    model_input_2 = Input(shape=(max_seq_len,), dtype='int32')
    model_input = [model_input_1, model_input_2]
    set_trainability(K, False)
    h = model_f(model_input)
    y_hat = model_g(h)
    l_hat = model_k(h)

    model_out = [y_hat, l_hat]

    model = Model(model_input, model_out)
    model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1, adv_weight],
                  optimizer=RMSprop(lr=lr, clipnorm=clipping))

    logger.info("Model ADN:")
    model.summary(print_fn=logger.info)

    return model, model_out


F, F_out = make_model_f(emb)
G, G_out = make_model_g()
K, K_out = make_model_k()
ADN, ADN_out = make_model_adn(F, G, K)

if not use_loaded_model:
    print("Pre-Training")
    if pred_weight != 0:
        H = F.predict([x1[train_idx], x2[train_idx]], batch_size=batch_size)
        set_trainability(K, True)
        K.fit(H, leaky_features[train_idx], epochs=10, batch_size=batch_size, sample_weight=sample_weight[train_idx])

    data_generator = DataGenerator(x1[train_idx], x2[train_idx], y[train_idx], leaky_features[train_idx],
                                   leaky_features_adv[train_idx], sample_weight[train_idx], batch_size=batch_size,
                                   shuffle=True, data_gen_mode=data_gen_mode)
    y_loss = []
    l_loss = []
    l_adv_loss = []

    lr_adn = lr
    lr_k = lr
    val_best = -1
    num_no_improv = 0
    best_epoch = -1
    for epoch in range(num_epochs):
        y_loss_batch = []
        l_loss_batch = []
        l_adv_loss_batch = []

        n_step = len(data_generator)
        for step in range(n_step):
            x1_batch, x2_batch, y_batch, l_batch, l_adv_batch, sw_batch = data_generator.next()

            if pred_weight != 0:
                set_trainability(K, True)
                H_batch = F.predict_on_batch([x1_batch, x2_batch])
                if use_sample_weight_train:
                    l_loss_batch.append(K.train_on_batch(H_batch, l_batch, sample_weight=sw_batch))
                else:
                    l_loss_batch.append(K.train_on_batch(H_batch, l_batch))
            else:
                l_loss_batch.append(0)

            set_trainability(K, False)
            if use_sample_weight_train:
                adn_loss = ADN.train_on_batch([x1_batch, x2_batch], [y_batch, l_adv_batch],
                                              sample_weight=[sw_batch, sw_batch])
            else:
                adn_loss = ADN.train_on_batch([x1_batch, x2_batch], [y_batch, l_adv_batch])
            y_loss_batch.append(adn_loss[1])
            l_adv_loss_batch.append(adn_loss[2])

            print('Epoch %2d/%2d. Step %3d%%. MAL: %.5f (y), %.5f (l), %.5f (l_adv)' %
                  (epoch, num_epochs, (step + 1) / n_step * 100,
                   float(np.mean(y_loss_batch)), float(np.mean(l_loss_batch)), float(np.mean(l_adv_loss_batch))),
                  end="\r")

        print("")
        y_loss.append(np.mean(y_loss_batch))
        l_loss.append(np.mean(l_loss_batch))
        l_adv_loss.append(np.mean(l_adv_loss_batch))

        y_val_preds = ADN.predict(([x1[val_idx], x2[val_idx]]), batch_size=2048)[0]
        y_val_preds = np.float64(y_val_preds)
        y_val = y[val_idx]
        score_val = dict()
        score_val['lss'] = log_loss(y_val, y_val_preds,
                                    sample_weight=sample_weight[val_idx] if use_sample_weight_train else None)
        score_val['acc'] = accuracy_score(y_val, y_val_preds > 0.5,
                                          sample_weight=sample_weight[val_idx] if use_sample_weight_train else None)
        score_val['auc'] = roc_auc_score(y_val, y_val_preds,
                                         sample_weight=sample_weight[val_idx] if use_sample_weight_train else None)
        logger.info(
            'Epoch %2d/%2d. Val Loss: %.5f, Val Acc: %.5f, Val Auc: %.5f, '
            'Train y Loss: %.5f, Train l Loss: %.5f, Train l_adv Loss: %.5f.' % (
                epoch, num_epochs, score_val['lss'], score_val['acc'], score_val['auc'],
                float(y_loss[-1]), float(l_loss[-1]), float(l_adv_loss[-1])))

        if val_best == -1 or (valid_std == 'lss' and score_val[valid_std] < val_best) or (
                valid_std != 'lss' and score_val[valid_std] > val_best):
            best_epoch = epoch
            val_best = score_val[valid_std]
            num_no_improv = 0
            ADN.save(file_adn)
            F.save(file_hidden_net)
            G.save(file_out_net)
            K.save(file_pred_net)
        else:
            if lr_adn >= 1e-8:
                lr_adn *= lr_decay
                lr_k *= lr_decay
                keras.backend.set_value(K.optimizer.lr, lr_k)
                keras.backend.set_value(ADN.optimizer.lr, lr_adn)
                print("LR_ADN/K reduced to %.8lf/%.8lf" % (lr_adn, lr_k))
            num_no_improv += 1
            if num_no_improv == max_no_improve:
                print("Early Stopping...")
                break
    y_loss, l_loss, l_adv_loss = y_loss[best_epoch], l_loss[best_epoch], l_adv_loss[best_epoch]
    logger.info('Best epoch\'s results on train set:')
    rec.add("train_y_loss", y_loss)
    rec.add("train_l_loss", l_loss)
    rec.add("train_l_adv_loss", l_adv_loss)

ADN = load_model(file_adn)
y_val_preds = np.float64(ADN.predict(([x1[val_idx], x2[val_idx]]), batch_size=2048)[0])
y_val = y[val_idx]
if use_sample_weight_train:
    loss_val = log_loss(y_val, y_val_preds, sample_weight=sample_weight[val_idx])
    acc_val = accuracy_score(y_val, y_val_preds > 0.5, sample_weight=sample_weight[val_idx])
    auc_val = roc_auc_score(y_val, y_val_preds, sample_weight=sample_weight[val_idx])
else:
    loss_val = log_loss(y_val, y_val_preds)
    acc_val = accuracy_score(y_val, y_val_preds > 0.5)
    auc_val = roc_auc_score(y_val, y_val_preds)
logger.info('Best results on val set:')
rec.add("val_loss", loss_val)
rec.add("val_acc", acc_val)
rec.add("val_auc", auc_val)

y_test_preds = np.float64(ADN.predict(([x1[test_idx], x2[test_idx]]), batch_size=2048)[0])
y_test = y[test_idx]

loss_weighted_test = log_loss(y_test, y_test_preds, sample_weight=sample_weight[test_idx])
acc_weighted_test = accuracy_score(y_test, y_test_preds > 0.5, sample_weight=sample_weight[test_idx])
auc_weighted_test = roc_auc_score(y_test, y_test_preds, sample_weight=sample_weight[test_idx])
loss_test = log_loss(y_test, y_test_preds)
acc_test = accuracy_score(y_test, y_test_preds > 0.5)
auc_test = roc_auc_score(y_test, y_test_preds)
logger.info('Best results on testing set:')
rec.add("test_loss", loss_test)
rec.add("test_acc", acc_test)
rec.add("test_auc", auc_test)
rec.add("test_weighted_loss", loss_weighted_test)
rec.add("test_weighted_acc", acc_weighted_test)
rec.add("test_weighted_auc", auc_weighted_test)

y_test_overlap = y_test[test_overlap_idx]
y_test_overlap_preds = y_test_preds[test_overlap_idx]
loss_weighted_test_overlap = log_loss(y_test_overlap, y_test_overlap_preds,
                                      sample_weight=sample_weight[test_idx][test_overlap_idx])
acc_weighted_test_overlap = accuracy_score(y_test_overlap, y_test_overlap_preds > 0.5,
                                           sample_weight=sample_weight[test_idx][test_overlap_idx])
auc_weighted_test_overlap = roc_auc_score(y_test_overlap, y_test_overlap_preds,
                                          sample_weight=sample_weight[test_idx][test_overlap_idx])
loss_test_overlap = log_loss(y_test_overlap, y_test_overlap_preds)
acc_test_overlap = accuracy_score(y_test_overlap, y_test_overlap_preds > 0.5)
auc_test_overlap = roc_auc_score(y_test_overlap, y_test_overlap_preds)
logger.info('Results on testing overlap set:')
rec.add("test_overlap_loss", loss_test_overlap)
rec.add("test_overlap_acc", acc_test_overlap)
rec.add("test_overlap_auc", auc_test_overlap)
rec.add("test_weighted_overlap_loss", loss_weighted_test_overlap)
rec.add("test_weighted_overlap_acc", acc_weighted_test_overlap)
rec.add("test_weighted_overlap_auc", auc_weighted_test_overlap)

y_test_no_overlap = y_test[test_no_overlap_idx]
y_test_no_overlap_preds = y_test_preds[test_no_overlap_idx]
loss_weighted_test_no_overlap = log_loss(y_test_no_overlap, y_test_no_overlap_preds,
                                         sample_weight=sample_weight[test_idx][test_no_overlap_idx])
acc_weighted_test_no_overlap = accuracy_score(y_test_no_overlap, y_test_no_overlap_preds > 0.5,
                                              sample_weight=sample_weight[test_idx][test_no_overlap_idx])
auc_weighted_test_no_overlap = roc_auc_score(y_test_no_overlap, y_test_no_overlap_preds,
                                             sample_weight=sample_weight[test_idx][test_no_overlap_idx])
loss_test_no_overlap = log_loss(y_test_no_overlap, y_test_no_overlap_preds)
acc_test_no_overlap = accuracy_score(y_test_no_overlap, y_test_no_overlap_preds > 0.5)
auc_test_no_overlap = roc_auc_score(y_test_no_overlap, y_test_no_overlap_preds)
logger.info('Results on testing no overlap set:')
rec.add("test_no_overlap_loss", loss_test_no_overlap)
rec.add("test_no_overlap_acc", acc_test_no_overlap)
rec.add("test_no_overlap_auc", auc_test_no_overlap)
rec.add("test_weighted_no_overlap_loss", loss_weighted_test_no_overlap)
rec.add("test_weighted_no_overlap_acc", acc_weighted_test_no_overlap)
rec.add("test_weighted_no_overlap_auc", auc_weighted_test_no_overlap)

y_test_same_preds = np.float64(ADN.predict(([x1[test_same_idx], x2[test_same_idx]]), batch_size=2048)[0])
y_test_same = y[test_same_idx]
acc_test_same = accuracy_score(y_test_same, y_test_same_preds > 0.5)
logger.info('Best results on testing same set:')
rec.add("test_same_acc", acc_test_same)

y_test_sick_preds = np.float64(ADN.predict(([x1[sick_idx], x2[sick_idx]]), batch_size=2048)[0])
y_test_sick = y[sick_idx]
loss_test_sick = log_loss(y_test_sick, y_test_sick_preds)
acc_test_sick = accuracy_score(y_test_sick, y_test_sick_preds > 0.5)
auc_test_sick = roc_auc_score(y_test_sick, y_test_sick_preds)
logger.info('Best results on testing set:')
rec.add("test_sick_loss", loss_test_sick)
rec.add("test_sick_acc", acc_test_sick)
rec.add("test_sick_auc", auc_test_sick)

y_test_msr_preds = np.float64(ADN.predict(([x1[msr_idx], x2[msr_idx]]), batch_size=2048)[0])
y_test_msr = y[msr_idx]
loss_test_msr = log_loss(y_test_msr, y_test_msr_preds)
acc_test_msr = accuracy_score(y_test_msr, y_test_msr_preds > 0.5)
auc_test_msr = roc_auc_score(y_test_msr, y_test_msr_preds)
logger.info('Best results on testing set:')
rec.add("test_msr_loss", loss_test_msr)
rec.add("test_msr_acc", acc_test_msr)
rec.add("test_msr_auc", auc_test_msr)

rec.save()

