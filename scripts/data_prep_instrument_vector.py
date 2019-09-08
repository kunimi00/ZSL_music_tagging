import pickle
import os
import json
import numpy as np

import sys
sys.path.append('../')

from util import fma_utils

'''
  From 163 genres, 
   we removed 6 due to their lack of instrument annotation, 
   -> leaving total of 157 genres. 

'''

'''
  Procedure 
  
  1) track by instrument (binary or confidence) vector
  2) track by genre vector
  3) genre by instrument (count or summed confidence) vector

'''



'''
  1) track x instrument vector  

'''


'''

  Open-MIC dataset

'''

DATA_ROOT = '/media/iu/openmic-2018'
OPENMIC = np.load(os.path.join(DATA_ROOT, 'openmic-2018.npz'))

with open(os.path.join(DATA_ROOT, 'class-map.json'), 'r') as f:
    class_map = json.load(f)

instrument_txt_list = []
for _k in list(class_map.keys()):
    instrument_txt_list.append(_k)
instrument_txt_list.sort()

X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']


CONFIDENCE_THRESHOLD = 0.5

song_key_by_inst_key_posneg_40bin_matrix = np.zeros((20000, 40))
song_key_by_inst_key_posneg_40conf_matrix = np.zeros((20000, 40))

for song_idx in range(20000):
    for inst_idx in range(20):
        if Y_mask[song_idx][inst_idx] == True:
            if Y_true[song_idx][inst_idx] > CONFIDENCE_THRESHOLD:
                song_key_by_inst_key_posneg_40bin_matrix[song_idx][inst_idx] = 1
                song_key_by_inst_key_posneg_40conf_matrix[song_idx][inst_idx] = Y_true[song_idx][inst_idx] - CONFIDENCE_THRESHOLD
            else:
                song_key_by_inst_key_posneg_40bin_matrix[song_idx][inst_idx+20] = 1
                song_key_by_inst_key_posneg_40conf_matrix[song_idx][inst_idx+20] = CONFIDENCE_THRESHOLD - Y_true[song_idx][inst_idx]



track_id_to_inst_posneg_40bin_dict = dict()
for idx in range(len(sample_key)):
    track_id_to_inst_posneg_40bin_dict[int(sample_key[idx].split('_')[0])] = song_key_by_inst_key_posneg_40bin_matrix[idx]


track_id_to_inst_posneg_40conf_dict = dict()
for idx in range(len(sample_key)):
    track_id_to_inst_posneg_40conf_dict[int(sample_key[idx].split('_')[0])] = song_key_by_inst_key_posneg_40conf_matrix[idx]


'''
  2) track x genre 
  
   : using FMA large dataset -> filter tracks with genre annotations

'''

tracks = fma_utils.load('/media/iu/fma_metadata/tracks.csv')
features = fma_utils.load('/media/iu/fma_metadata/features.csv')
echonest = fma_utils.load('/media/iu/fma_metadata/echonest.csv')

genres = fma_utils.load('/media/iu/fma_metadata/genres.csv')

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

print(tracks.shape, features.shape, echonest.shape)

track_ids = tracks.index

tracks_large = tracks['set', 'subset'] <= 'large'
track_genres_top = tracks.loc[tracks_large, ('track', 'genre_top')].values.tolist()
track_genres = tracks.loc[tracks_large, ('track', 'genres')].values.tolist()
track_ids = tracks.loc[tracks_large].index
genre_titles = genres['title'].tolist()
genre_ids = genres.index.tolist()

track_ids_with_genres = []
track_id_to_genre_id_dict = {}
for i in range(len(track_genres)):
    if len(track_genres[i]) > 0:
        track_ids_with_genres.append(track_ids[i])
        track_id_to_genre_id_dict[track_ids[i]] = track_genres[i]
    else:
        continue

print('track_ids_with_genres', len(track_ids_with_genres)) # 104343

genre_ids.sort()

track_ids_with_inst = []
for _key in sample_key:
    track_ids_with_inst.append(int(_key.split('_')[0]))

print('track_ids_with_inst', len(track_ids_with_inst)) # 20000

# Here we used prefiltered tracks (19466) and genres (157)
prefiltered_track_ids_in_key_order = pickle.load(open('data_common/fma/track_ids_in_key_order.p', 'rb'))
prefiltered_tag_ids_in_key_order = pickle.load(open('data_common/fma/tag_ids_in_key_order.p', 'rb'))

track_key_to_genre_key_binary_matrix = []

for key, t_id in enumerate(prefiltered_track_ids_in_key_order):
    curr_binary = np.zeros(len(prefiltered_tag_ids_in_key_order))
    for curr_genre_id in track_id_to_genre_id_dict[t_id]:
        curr_binary[prefiltered_tag_ids_in_key_order.index(curr_genre_id)] = 1

    track_key_to_genre_key_binary_matrix.append(curr_binary)

track_key_to_genre_key_binary_matrix = np.array(track_key_to_genre_key_binary_matrix)

print('track_key_to_genre_key_binary_matrix shape ', track_key_to_genre_key_binary_matrix.shape)
# (19466, 157)
genre_key_to_track_key_binary_matrix = track_key_to_genre_key_binary_matrix.T
# (157, 19466)




'''
  3) genre x inst
    (save as id to vector dictionary)
'''

# track_id_to_inst_posneg_40bin_dict / track_id_to_inst_posneg_40conf_dict


genre_id_to_inst_posneg40_cnt_dict = {}
genre_id_to_inst_posneg40_conf_dict = {}

for genre_key in range(len(prefiltered_tag_ids_in_key_order)):
    genre_id = prefiltered_tag_ids_in_key_order[genre_key]

    genre_id_to_inst_posneg40_cnt_dict[genre_id] = np.zeros((40,))
    genre_id_to_inst_posneg40_conf_dict[genre_id] = np.zeros((40,))

    curr_track_keys = np.argwhere(genre_key_to_track_key_binary_matrix[genre_key] == 1).squeeze()

    for _track_key in curr_track_keys:
        _track_id = prefiltered_track_ids_in_key_order[_track_key]
        _curr_track_inst_bin = track_id_to_inst_posneg_40bin_dict[_track_id]
        _curr_track_inst_conf = track_id_to_inst_posneg_40conf_dict[_track_id]

        genre_id_to_inst_posneg40_cnt_dict[genre_id] += _curr_track_inst_bin
        genre_id_to_inst_posneg40_conf_dict[genre_id] += _curr_track_inst_conf

'''
  Standardization along genre dimension 
'''

genre_id_to_inst_posneg40_cnt_norm_dict = {}
genre_id_to_inst_posneg40_conf_norm_dict = {}

for genre_key in range(len(prefiltered_tag_ids_in_key_order)):
    genre_id = prefiltered_tag_ids_in_key_order[genre_key]

    curr_genre_vector = genre_id_to_inst_posneg40_cnt_dict[genre_id]
    _mean = curr_genre_vector.mean()
    _std = curr_genre_vector.std()

    if _std == 0:
        print("Error normalizing ! (shouldn't happend since using pre-filtered tags / tracks)", genre_id, 'cnt')
        exit(0)
    curr_genre_vector_norm = (curr_genre_vector - _mean) / _std
    genre_id_to_inst_posneg40_cnt_norm_dict[genre_id] = curr_genre_vector_norm

    curr_genre_vector = genre_id_to_inst_posneg40_conf_dict[genre_id]
    _mean = curr_genre_vector.mean()
    _std = curr_genre_vector.std()

    if _std == 0:
        print("Error normalizing ! (shouldn't happend since using pre-filtered tags / tracks)", genre_id, 'conf')
        exit(0)
    curr_genre_vector_norm = (curr_genre_vector - _mean) / _std
    genre_id_to_inst_posneg40_conf_norm_dict[genre_id] = curr_genre_vector_norm



savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_cnt_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_cnt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')

savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_conf_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_conf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')

savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_cnt_norm_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_cnt_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')

savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_conf_norm_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_conf_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')









