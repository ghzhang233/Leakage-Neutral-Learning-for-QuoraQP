import logging
import sys
import threading
import numpy as np


class DataGenerator:

    def __init__(self, x1, x2, y, le, l_adv, sample_weight, batch_size=32, shuffle=True, data_gen_mode="normal"):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_gen_mode = data_gen_mode
        self.n = len(x1)

        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.le = le
        self.l_adv = l_adv
        self.sample_weight = sample_weight
        if data_gen_mode == 'prob':
            self.sample_weight /= np.sum(self.sample_weight)

        self.batch_index = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
        self.epoch_end_flag = False

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.batch_index == 0:
                self._set_index_array()
            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
                self.epoch_end_flag = True
            yield self.index_array[current_index: current_index + self.batch_size]
            if self.epoch_end_flag:
                self.on_epoch_end()
                self.epoch_end_flag = False

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def _get_batches_of_samples(self, index_array):
        x1_batch = self.x1[index_array]
        x2_batch = self.x2[index_array]
        y_batch = self.y[index_array]
        le_batch = self.le[index_array]
        l_adv_batch = self.l_adv[index_array]
        if self.data_gen_mode == 'normal':
            sample_weight_batch = self.sample_weight[index_array]
        elif self.data_gen_mode == 'prob':
            sample_weight_batch = np.ones(self.batch_size)
        else:
            raise NotImplementedError

        return x1_batch, x2_batch, y_batch, le_batch, l_adv_batch, sample_weight_batch

    def next(self):
        with self.lock:
            if self.data_gen_mode == 'normal':
                index_array = next(self.index_generator)
            elif self.data_gen_mode == 'prob':
                index_array = np.random.choice(a=self.n, size=self.batch_size, p=self.sample_weight)
            else:
                raise NotImplementedError

        return self._get_batches_of_samples(index_array)


def get_logger(file_log):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(file_log)
    file_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


class ResultRecorder:
    def __init__(self, logger, file_result):
        self.logger = logger
        self.ret = dict()
        self.file_result = file_result

    def add(self, name, value, show=True):
        if name in self.ret.keys():
            print("Key %s Exist! Error!" % name)
            return
        self.ret[name] = value
        if show:
            if type(value) == float:
                if value >= 1e-5:
                    self.logger.info("%s: %.5lf" % (name, value))
                else:
                    self.logger.info("%s: %lf" % (name, value))
            else:
                self.logger.info("%s: %s" % (name, str(value)))

    def save(self):
        with open(self.file_result, "a", encoding="utf-8") as fin:
            ret_str = "\t".join([k + ": " + str(self.ret[k]) for k in sorted(self.ret.keys())]) + "\n"
            fin.write(ret_str)

