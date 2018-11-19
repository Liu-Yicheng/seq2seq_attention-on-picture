import config as cfg
import os
import numpy as np
import random

class Dataset():
    def __init__(self):
        self.label_txt = cfg.Label_txt
        self.class_to_ind = dict(zip(cfg.Class, range(0, len(cfg.Class))))

        self.train_list, self.val_list = self.get_trainval_list()
        self.labels = self.get_label()

        self.train_step = len(self.train_list)
        self.val_num = len(self.val_list)
        self.val_index = 0
        self.val_finish = False

    def  get_trainval_list(self):

        with open(cfg.Train_list_txt, 'r') as train_f:
            lines = train_f.readlines()
        train_list = [line.strip('\n') for line in lines]

        with open(cfg.Val_list_txt, 'r') as val_f:
            lines = val_f.readlines()
        val_list = [line.strip('\n') for line in lines]

        return train_list, val_list

    def get_label(self):
        labels = {}
        with open(cfg.Label_txt, 'r') as f:
            lines =f.readlines()
            for line in lines:
                index = line.split('\t')[0].strip('\n')
                class_name = line.split('\t')[1].strip('\n')
                label = self.class_to_ind[class_name]
                labels[index] = label
        return labels


    def get_train_batch(self):

        input_sequence = np.zeros((cfg.Input_seq_len, cfg.Batch_size, cfg.Input_dim), dtype=float)
        label_sequence = np.zeros((cfg.Input_seq_len + 1, cfg.Batch_size), dtype=int)
        choose_index = random.sample(range(len(self.train_list)-cfg.Input_seq_len), cfg.Batch_size)
        for i, choose in enumerate(choose_index):
            choose_list = self.train_list[choose : choose+cfg.Input_seq_len]
            for index, single_index  in enumerate(choose_list):
                npy_path = os.path.join(cfg.Npy_dir, single_index+'.npy')
                sequence = np.load(npy_path)
                input_sequence[index, i, :] = sequence
                label_sequence[index+1, i] = self.labels[single_index]
        return input_sequence, label_sequence

    def get_val_batch(self):
        input_sequence = np.zeros((cfg.Input_seq_len, cfg.Batch_size, cfg.Input_dim), dtype=float)
        label_sequence = np.zeros((cfg.Input_seq_len+1, cfg.Batch_size), dtype=int)
        file_index = []
        for i in range(cfg.Input_seq_len):
            file_index.append([])
        for i in range(cfg.Batch_size):
            self.val_index += 1
            if self.val_index > self.val_num - cfg.Input_seq_len:
                self.val_index -= 1
                self.val_finish = True

            choose_list = self.val_list[self.val_index : self.val_index+cfg.Input_seq_len]
            for index, single_index  in enumerate(choose_list):
                npy_path = os.path.join(cfg.Npy_dir, single_index+'.npy')
                sequence = np.load(npy_path)
                input_sequence[index, i, :] = sequence
                label_sequence[index+1, i] = self.labels[single_index]
                file_index[index].append(single_index)
        return file_index, input_sequence, label_sequence
