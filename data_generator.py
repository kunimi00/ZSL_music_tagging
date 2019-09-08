import numpy as np
import keras
import random
import pickle
import threading
import os
from tqdm import tqdm
from datetime import datetime

'''

 What we need to do :

 (1) prepare matrix
  given set of tags and tracks,
  prepare a binanry matrix (given track (A, B, or AB) x given tag (training tags) 
  
 (2) feed data batch (anchor audio mel, pos tag vector, neg tag vectors)

'''

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, args, curr_phase, shuffle=True):

        self.batch_size = args.batch_size
        self.num_negative_sampling = args.num_negative_sampling
        self.global_mel_mean = args.global_mel_mean
        self.global_mel_std = args.global_mel_std
        self.dir_mel = args.dir_mel
        self.num_frame = args.num_frame
        self.dataset = args.dataset

        self.track_key_to_tag_binary_matrix = args.track_key_to_tag_binary_matrix
        self.tag_key_to_track_binary_matrix = args.tag_key_to_track_binary_matrix
        self.tag_id_to_w2v_vector_dict  = args.tag_id_to_w2v_vector_dict

        self.track_ids_in_key_order = args.track_ids_in_key_order
        self.tag_ids_in_key_order = args.tag_ids_in_key_order

        self.track_id_to_file_path_dict = args.track_id_to_file_path_dict


        # (1) load track split
        print('loading ' + args.track_split + ' track splits..')
        savename = args.data_common_path + '/track_keys_' + args.track_split + '_' + args.track_split_name + '_' + args.tag_split_name + '.p'

        if args.track_split in ['AB', 'A', 'B']:
            track_keys_train, track_keys_valid = pickle.load(open(savename, 'rb'))

            if curr_phase == 'train':
                print('Preparing TRAIN tracks..')
                self.curr_phase_track_keys = track_keys_train
                print('using training set tracks :', len(self.curr_phase_track_keys))

            elif curr_phase == 'valid':
                print('Preparing VALID tracks..')
                self.curr_phase_track_keys = track_keys_valid
                print('using valid set tracks :', len(self.curr_phase_track_keys))

        else:
            print('ERROR : Training is only done with AB, A, or B.')
            exit(0)

        # (2) load tag split
        self.curr_phase_tag_keys = args.train_tag_keys


        '''
        
        Things to prepare : 
        
            - self.curr_phase_tag_keys : shuffled tag keys
            - self.curr_phase_track_keys : shuffled track keys  
            
            - self.curr_tag_by_curr_track_matrix
            - self.curr_track_by_curr_tag_matrix
            
        '''

        # Filter the whole matrix into curr phase matrix
        curr_track_by_tag_matrix = self.track_key_to_tag_binary_matrix[self.curr_phase_track_keys]

        self.curr_tag_by_curr_track_matrix = curr_track_by_tag_matrix.T[self.curr_phase_tag_keys]
        self.curr_track_by_curr_tag_matrix = self.curr_tag_by_curr_track_matrix.T

        print('curr_track_by_curr_tag_matrix :', self.curr_track_by_curr_tag_matrix.shape)

        self.track_key_to_curr_phase_tag_idx_binary_dict = {}
        for _idx in range(len(self.curr_phase_track_keys)):
            curr_tr_key = self.curr_phase_track_keys[_idx]
            self.track_key_to_curr_phase_tag_idx_binary_dict[curr_tr_key] = \
                self.curr_track_by_curr_tag_matrix[_idx]

        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.curr_phase_track_keys) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_curr_keys = [self.curr_phase_track_keys[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_curr_keys)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.curr_phase_track_keys))
        if self.shuffle == True:
            # print('\n shuffled dataset.')
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_curr_keys):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # initializing
        col_anchor_audio = []
        col_pos_item = []
        col_neg_items = [[] for j in range(self.num_negative_sampling)]
        y_batch = []

        for i, key in enumerate(list_curr_keys):

            mel_path = self.track_id_to_file_path_dict[self.track_ids_in_key_order[key]]
            bin_list = self.track_key_to_curr_phase_tag_idx_binary_dict[key]

            # load anchor audio
            anchor_audio = np.load(os.path.join(self.dir_mel, mel_path.replace('.mp3', '.npy')))

            while anchor_audio.shape[0] < self.num_frame:
                print('\n error audio : ' + mel_path)
                anchor_audio = np.concatenate((anchor_audio, anchor_audio), axis=0)
            else:
                audio_chunk_begin = random.randint(0, anchor_audio.shape[0] - self.num_frame)

            anchor_audio = anchor_audio[audio_chunk_begin: audio_chunk_begin + self.num_frame]

            anchor_audio -= self.global_mel_mean
            anchor_audio /= self.global_mel_std

            # load pos/neg item
            curr_pos_indices = [i for i, x in enumerate(bin_list) if x == 1]
            curr_neg_indices = [i for i, x in enumerate(bin_list) if x == 0]

            picked_pos_idx = random.choice(curr_pos_indices)
            picked_neg_indices = random.sample(curr_neg_indices, self.num_negative_sampling)

            pos_item = self.tag_id_to_w2v_vector_dict[self.tag_ids_in_key_order[self.curr_phase_tag_keys[picked_pos_idx]]]
            neg_items = [self.tag_id_to_w2v_vector_dict[self.tag_ids_in_key_order[self.curr_phase_tag_keys[i]]] for i in
                         picked_neg_indices]

            # add to curr batch
            col_anchor_audio.append(anchor_audio)
            col_pos_item.append(pos_item)
            for neg_iter in range(self.num_negative_sampling):
                col_neg_items[neg_iter].append(neg_items[neg_iter])

            # blank labels
            label = np.zeros((self.num_negative_sampling + 1))
            label[0] = 1
            y_batch.append(list(label))

        col_anchor_audio = np.array(col_anchor_audio)
        col_pos_item = np.array(col_pos_item)
        for j in range(self.num_negative_sampling):
            col_neg_items[j] = np.array(col_neg_items[j])

        x_batch = [col_anchor_audio, col_pos_item] + [col_neg_items[j] for j in range(self.num_negative_sampling)]
        y_batch = np.array(y_batch)

        return x_batch, y_batch
