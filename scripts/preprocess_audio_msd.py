# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os, librosa, argparse, pickle, multiprocessing
import numpy as np
from multiprocessing import Process, Queue
import pickle
from tqdm import tqdm
import pathlib

'''
Options 

 --dir_wav : path where MSD mp3 files are present 
 --dir_mel : path to save mel files

'''


def preprocess_audio(args, type_feature, metalist, q):

    pathlib.Path(args.dir_mel).mkdir(parents=True, exist_ok=True)
    filtered_list = []
    error_list = []

    print('process running..')

    for el_idx in range(len(metalist)):

        element = metalist[el_idx]
        path = os.path.join(args.dir_wav, element)
        curr_npy_path = os.path.join(args.dir_mel, element[:-4])

        if os.path.isfile(curr_npy_path + '.npy'):
            filtered_list.append(element)
            continue

        else:
            try:
                if type_feature.endswith('linear') or type_feature.endswith('mel'):
                    audio, _ = librosa.load(path, sr=args.sample_rate, duration=30, mono=True)

                    if audio.shape[0] < (args.sample_rate * 29):
                        print('\n * error short file (less than 29 sec): ' + str(element) + ': ' + str(audio.shape))
                        error_list.append(element)
                        continue

                    ## STFT
                    D = librosa.core.stft(audio,
                                     n_fft=args.fft_size,
                                     win_length=args.window,
                                     window='hann',
                                     hop_length=args.hop)
                    y = np.abs(D)  # H x T

                    # get mel-spec or lin-spec
                    if type_feature.endswith('mel'):
                        ## apply mel filter
                        # if args.aug_mel == 1:
                        #     mel_basis_low = librosa.filters.mel(args.sample_rate,
                        #                                     args.n_fft,
                        #                                     n_mels=args.mel_dim,
                        #                                     fmin=args.fmin,
                        #                                     fmax=2000)
                        #     y_low = np.dot(mel_basis_low, y)
                        mel_basis = librosa.filters.mel(args.sample_rate,
                                                        n_fft=args.fft_size,
                                                        n_mels=args.melBin)

                        y = np.dot(mel_basis, y)
                        # ref_level_db = args.ref_level_db_mel

                    # else:
                        # ref_level_db = args.ref_level_db_lin

                    ## normalize and clip: This finally yields spectrogram amplitude between [1e-5, 10]
                    # y = 20 * np.log10(np.maximum(1e-5, y)) - ref_level_db
                    # y = np.clip(-(y - args.min_level_db) / args.min_level_db, 0, 1)

                    ## simpler normalization
                    y = np.log10(1 + 10 * y)

                    y = y.astype(np.float32)

                    ## augment mel spectrogram for low frequency
                    ## normalize and clip for low freq: This finally yields spectrogram amplitude between [1e-5, 10]
                    # if args.aug_mel == 1 and type_feature.endswith('mel'):
                    #     y_low = 20 * np.log10(np.maximum(1e-5, y_low)) - ref_level_db
                    #     y_low = np.clip(-(y_low - args.min_level_db) / args.min_level_db, 0, 1)
                    #     y = np.concatenate((y, y_low), axis=0)
                    y = y.T  # H x T -> T x H (time x freq)

                pathlib.Path('/'.join(curr_npy_path.split('/')[:-1]) + '/').mkdir(parents=True, exist_ok=True)
                np.save(curr_npy_path, y)
                print('  => processed..' + curr_npy_path + ' / ' + str(y.dtype) + ' / ' + str(y.shape))

                filtered_list.append(element)

            except EOFError as error:
                print('EOFError :', element)
                error_list.append(element)
                continue

    q.put((type_feature, filtered_list, error_list))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='script')

        parser.add_argument('--dir_wav', type=str, default='/media/bach2/dataset/MSD/songs',
                            help='path to dataset')
        parser.add_argument('--dir_mel', type=str, default='/media/irene/dataset/MSD/mel128',
                            help='path to save separate mel data files')

        parser.add_argument('--num_part', type=int, default=12, help='num of cpu')

        args = parser.parse_args()

        args.min_level_db = -100
        args.ref_level_db_mel = 20 + 0.2 * args.min_level_db  # adjust to fit in [0,1]
        args.ref_level_db_lin = 20
        # args.preemphasis_ratio = 0.97

        args.fft_size = 1024
        args.window = 1024
        args.hop = 512
        args.melBin = 128
        args.sample_rate = 22050


        print('Dataset to preprocess: MSD')
        path_list_dict = pickle.load(open('../data_common/msd/track_id_to_file_path_dict.p', 'rb'))

        metalist = path_list_dict.values()
        metalist.sort()

        num_part = args.num_part
        part = int(len(metalist)/num_part)
        metalist_list = [metalist[part*i:part*(i+1)] for i in range(num_part-1)]
        metalist_list.append(metalist[part*(num_part-1):])

        print(len(metalist_list))

        q = Queue()

        p_mels = []
        for i in range(num_part):
            p_mels.append(Process(target=preprocess_audio, args=(args, 'mel', metalist_list[i], q)))

        for i in range(num_part):
            p_mels[i].daemon = True
            p_mels[i].start()

        filtered_list = []
        error_list = []

        for _ in range(num_part):
            _, _filtered, _error = q.get()
            filtered_list.extend(_filtered)
            error_list.extend(_error)

        filtered_list.sort()
        print(len(filtered_list))

        with open('MSD_audio_filtered_track_file_path_list.p', 'wb') as handle:
            pickle.dump(filtered_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('MSD_audio_error_list.p', 'wb') as handle:
            pickle.dump(error_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for i in range(num_part):
            p_mels[i].join()

        print(error_list)


    finally:
        for p in multiprocessing.active_children():
            p.terminate()
