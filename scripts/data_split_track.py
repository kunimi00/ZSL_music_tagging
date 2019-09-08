'''
INFO : keys, ids

Tag
 - ID : textual information
 - key : unique number key assigned to each tag
 - idx : index on the binary matrix

Track
 - ID : track unique ID
 - key : key (index assigned after filtering) - necessary for different shuffling
 - idx : index on the binary matrix

tag_key_to_track_key_binary_matrix (full tag x full track)
  : both dimension in key order (after shuffle)


What THIS file does:

 1. Given
    - a certain split set of tags (--tag_split_name *)
    - track to tag binary matrix
     (ttr_glove_track_key_to_tag_binary_dict.p / ttr_glove_track_to_tag_bin_matrix.p)

 -> split tracks into A / B / (AB) / C set

 2. For each of track subsets for training (A, B, AB),
 split data into training - valid set. (TGS01, TGS02..)

    Randomized Training / Valid track splits for each combination of
     -> trag subset + track split + tag split
    will be saved.

  : ex. AB + TRS01 + TGS02  ->  track_keys_AB_TRS01_TGS02.csv
        A + TRS02 + TGS02 ->  track_keys_A_TRS02_TGS02.csv

 A, B, C, AB split will be saved as a separate file.

'''
import numpy as np
import random
import pickle
import os
from tqdm import tqdm
import argparse
import sys
sys.path.append('../')


def main():
    parser = argparse.ArgumentParser(description='training script')

    # Dataset
    parser.add_argument('--dataset', type=str, default='',
                        help='dataset to work on : msd / fma')


    # Get tag split from pre-store file. ex. TGS02
    parser.add_argument('--tag_split_name', type=str, default='TGS01',
                        help='Tag split name.')

    # Name this track split. ex. TRS01
    parser.add_argument('--track_split_name', type=str, default='TRS01',
                        help='Track split name.')


    args = parser.parse_args()

    data_common_path = os.path.join('data_common', args.dataset)

    # Load data
    tag_key_to_track_binary_matrix = pickle.load(open(data_common_path + '/all_tag_to_track_bin_matrix.p', 'rb'))
    track_ids_in_order = pickle.load(open(data_common_path + '/track_ids_in_key_order.p', 'rb'))

    all_track_keys = range(len(track_ids_in_order))
    tag_split_file_name = data_common_path + '/tag_key_split_' + str(args.tag_split_name) + '.p'

    train_tag_keys, test_tag_keys = pickle.load(open(tag_split_file_name, 'rb'))

    '''
    <Splitting Procedure>
    
    Given tag list of training (X) / test set (Y), 
    we need to split tracks into 3 groups A / B / C
        
        
        1) First, we collect all tracks that are annotated with X at least once : A + B
         + split data into training - valid set. (TGS01, TGS02..)
        
        2) We automatically get the other part C
        
        3) From A+B, we collect all tracks that are annotated with Y : B
         + split data into training - valid set. (TGS01, TGS02..)
        
        4) We automatically get the other part A
         + split data into training - valid set. (TGS01, TGS02..)

    '''

    '''
    
    1. Getting (A + B) split : ALL tracks containing training tags (some contain test tags)
    
    '''

    print('1) First, we collect all tracks that are annotated with X at least once : A + B')
    print(' 1-1) create X by All track matrix first.')

    all_track_by_train_tag_matrix = tag_key_to_track_binary_matrix[train_tag_keys].T

    print(' 1-2) filter out by tag occurrence - We automatically get the other part (C).')

    filtered_track_keys_AB = []
    _err_exceptional = []
    leftover_track_keys_C = []
    for tr_key in tqdm(all_track_keys):
        if 1 in set(all_track_by_train_tag_matrix[tr_key]):
            if 0 in set(all_track_by_train_tag_matrix[tr_key]):
                filtered_track_keys_AB.append(all_track_keys[tr_key])
            else:
                print('exceptional case..!', all_track_keys[tr_key])
                _err_exceptional.append(all_track_keys[tr_key])
                continue
        else:
            leftover_track_keys_C.append(all_track_keys[tr_key])
            continue

    '''
    
    ex. file name : track_keys_AB_TRS01_TGS02.csv
    
    '''

    print(' Number of all filtered_track_keys (A + B) :', len(filtered_track_keys_AB))

    random.shuffle(filtered_track_keys_AB)
    num_whole_set = len(filtered_track_keys_AB)
    num_train = int(num_whole_set * 0.9)

    AB_track_keys_train = filtered_track_keys_AB[:num_train]
    AB_track_keys_valid = filtered_track_keys_AB[num_train:]

    print(' TRAIN SET filtered_track_keys (A + B) :', len(AB_track_keys_train))
    print(' VALID SET filtered_track_keys (A + B) :', len(AB_track_keys_valid))

    savename = data_common_path + '/track_keys_AB_' + args.track_split_name + '_' + args.tag_split_name + '.p'
    with open(savename, 'wb') as handle:
        pickle.dump([AB_track_keys_train, AB_track_keys_valid], handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    
    2. We automatically get the other part (C).
    
    '''
    print('We automatically get the other part (C).')
    print(' Number of all filtered_track_keys (C) :', len(leftover_track_keys_C))
    savename = data_common_path + '/track_keys_C_' + args.track_split_name + '_' + args.tag_split_name + '.p'
    with open(savename, 'wb') as handle:
        pickle.dump(leftover_track_keys_C, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    
    3. Getting (B) split : tracks containing BOTH training tags AND test tags
       From A+B, we collect all tracks that are annotated with Y : B
    
    '''
    print('3) From A+B, we collect all tracks that are annotated with Y : B')
    print(' 3-1) create Y by All track matrix first.')

    all_track_by_test_tag_matrix = tag_key_to_track_binary_matrix[test_tag_keys].T

    filtered_track_keys_B = []
    leftover_track_keys_A = []
    for tr_key in tqdm(filtered_track_keys_AB):
        # If 1 exists in both train and test tags, ..
        if 1 in set(all_track_by_train_tag_matrix[tr_key]):
            if 1 in set(all_track_by_test_tag_matrix[tr_key]):
                filtered_track_keys_B.append(tr_key)
            else:
                leftover_track_keys_A.append(tr_key)
                continue
        else:
            print('something is wrong creating B split from AB', tr_key)
            exit(0)
            continue

    print(' Number of all filtered_track_keys (B) :', len(filtered_track_keys_B))

    random.shuffle(filtered_track_keys_B)
    num_whole_set = len(filtered_track_keys_B)
    num_train = int(num_whole_set * 0.9)

    B_track_keys_train = filtered_track_keys_B[:num_train]
    B_track_keys_valid = filtered_track_keys_B[num_train:]

    print(' TRAIN SET filtered_track_keys (B) :', len(B_track_keys_train))
    print(' VALID SET filtered_track_keys (B) :', len(B_track_keys_valid))

    savename = data_common_path + '/track_keys_B_' + args.track_split_name + '_' + args.tag_split_name + '.p'
    with open(savename, 'wb') as handle:
        pickle.dump([B_track_keys_train, B_track_keys_valid], handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    
    4. We automatically get the other part (A).

    '''
    print('4) We automatically get the other part (A).')
    print(' Number of all filtered_track_keys (A) :', len(leftover_track_keys_A))

    random.shuffle(leftover_track_keys_A)
    num_whole_set = len(leftover_track_keys_A)
    num_train = int(num_whole_set * 0.9)

    A_track_keys_train = leftover_track_keys_A[:num_train]
    A_track_keys_valid = leftover_track_keys_A[num_train:]

    print(' TRAIN SET filtered_track_keys (A) :', len(A_track_keys_train))
    print(' VALID SET filtered_track_keys (A) :', len(A_track_keys_valid))

    savename = data_common_path + '/track_keys_A_' + args.track_split_name + '_' + args.tag_split_name + '.p'
    with open(savename, 'wb') as handle:
        pickle.dump([A_track_keys_train, A_track_keys_valid], handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    Just Checking
    '''
    print('- Rechecking each matrix..')
    AB_track_by_train_tag_matrix = all_track_by_train_tag_matrix[filtered_track_keys_AB]
    A_track_by_train_tag_matrix = all_track_by_train_tag_matrix[leftover_track_keys_A]
    B_track_by_train_tag_matrix = all_track_by_train_tag_matrix[filtered_track_keys_B]
    B_track_by_test_tag_matrix = all_track_by_test_tag_matrix[filtered_track_keys_B]
    C_track_by_train_tag_matrix = all_track_by_test_tag_matrix[leftover_track_keys_C]

    matrix_arr = [AB_track_by_train_tag_matrix, A_track_by_train_tag_matrix, B_track_by_train_tag_matrix,
                  B_track_by_test_tag_matrix, C_track_by_train_tag_matrix]

    for midx in range(len(matrix_arr)):
        mat = matrix_arr[midx]
        for idx in range(mat.shape[0]):
            if 1 not in set(mat[idx]):
                print('err..!',  midx, idx)

    print('done.')

if __name__ == '__main__':
    main()






