'''
INFO : keys, ids

1. tag
 - key : index from the (pre-filtered) list (1126 tags)
 - id : original textual representation (ex. '00s', 'rock')

2. track
 - key : index from the (pre-filtered) list (? tracks)
 - id : original track ID given in MSD


What THIS file does:

 Randomly split tags into Training / Test set.
 (We're not using validation set for tags.)

 - Tag keys are given in sorted order of the whole set of keys (FIXED).
 - Randomized split will be represented as set of keys.
 - Each TAG split should be named by 'TGS' + a number.
   ex. TGS01, TGS02
 - Returned result is going to be a list of two lists (training keys, test keys).
 - An example of output file name will be :
   ex. tag_key_split_TGS01.csv

'''

import numpy as np
import pickle
import random
import argparse
import os
import sys
sys.path.append('../')

def main():
    parser = argparse.ArgumentParser(description='training script')

    parser.add_argument('--dataset', type=str, default='',
                        help='dataset to work on : msd / fma')

    parser.add_argument('--tag_split_name', type=str, default='',
                        help='ex. TGS01, TGS02')

    args = parser.parse_args()

    if args.dataset == '':
        print('Dataset should be specified.')
        exit(0)

    data_common_path = os.path.join('data_common', args.dataset)

    if args.tag_split_name == '':
        print('Proper tag split name should be given. (ex. TGS01, TGS02..)')
        exit(0)

    else:
        print('Splitting tags into training / test set..')
        tag_ids_in_key_order = pickle.load(open(data_common_path + '/tag_ids_in_key_order.p', 'rb'))
        tag_all_keys = list(range(len(tag_ids_in_key_order)))
        random.shuffle(tag_all_keys)

        num_whole_set = len(tag_all_keys)
        num_train = int(num_whole_set * 0.8)

        tag_keys_train = tag_all_keys[:num_train]
        tag_keys_test = tag_all_keys[num_train:]

        savename = data_common_path + '/tag_key_split_' + str(args.tag_split_name) + '.p'

        with open(savename, 'wb') as handle:
            pickle.dump([tag_keys_train, tag_keys_test], handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(savename + ' : has been saved')


if __name__ == '__main__':
    main()
