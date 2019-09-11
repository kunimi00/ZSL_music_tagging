# -*- coding: utf-8 -*-
import os, librosa, argparse, pickle, multiprocessing
import numpy as np
from multiprocessing import Process, Queue
# from scipy import signal
import pickle
from tqdm import tqdm
import pathlib

'''
Options 

 --dir_wav : path where FMA wav files are present 
 --dir_mel : path to save mel files

'''


def save_obj_curr_folder(obj, name):
    with open(name + '.p', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_curr_folder(name):
    with open(name + '.p', 'rb') as f:
        return pickle.load(f)


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def get_mel_path(mel_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(mel_dir, tid_str[:3], tid_str + '.mel')


def preprocess_audio(args, type_feature, metalist, q):

    pathlib.Path(args.dir_mel).mkdir(parents=True, exist_ok=True)
    # print(args.dir_mel)
    ## params for stft
    # window_len = int(np.ceil(args.frame_len_inMS * args.sample_rate / 1000))
    # hop_length = int(np.ceil(args.frame_shift_inMS * args.sample_rate / 1000))
    # target_list = []

    filtered_list = []
    error_list = []

    print('process running..')

    # for element in tqdm(metalist):
    for element in metalist:

        path = get_audio_path(args.dir_wav, element)
        curr_npy_path = get_mel_path(args.dir_mel, element)

        # print('processing..' + curr_npy_path)
        if os.path.isfile(curr_npy_path + '.npy'):
            # print('skipped.')
            continue
        else:
            if type_feature.endswith('linear') or type_feature.endswith('mel'):
                audio, _ = librosa.load(path, sr=args.sample_rate, mono=True)

                if audio.shape[0] < 3000:
                    print('\n * error file : ' + str(element))
                    print('audio length: ' + str(audio.shape[0]))
                    error_list.append(element)
                    continue

                ## STFT
                D = librosa.stft(audio,
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
                                                    args.fft_size,
                                                    n_mels=args.melBin)

                    y = np.dot(mel_basis, y)
                    ref_level_db = args.ref_level_db_mel
                else:
                    ref_level_db = args.ref_level_db_lin

                ## normalize and clip: This finally yields spectrogram amplitude between [1e-5, 10]
                # y = 20 * np.log10(np.maximum(1e-5, y)) - ref_level_db
                # y = np.clip(-(y - args.min_level_db) / args.min_level_db, 0, 1)

                ## simpler normalization
                y = np.log10(1 + 10 * y)

                ## augment mel spectrogram for low frequency
                ## normalize and clip for low freq: This finally yields spectrogram amplitude between [1e-5, 10]
                # if args.aug_mel == 1 and type_feature.endswith('mel'):
                #     y_low = 20 * np.log10(np.maximum(1e-5, y_low)) - ref_level_db
                #     y_low = np.clip(-(y_low - args.min_level_db) / args.min_level_db, 0, 1)
                #     y = np.concatenate((y, y_low), axis=0)
                y = y.T  # H x T -> T x H (time x freq)


            # also save npy for the future
            # curr_npy_path = os.path.join(args.dir_mel, '/'.join(element.split('/')[-4:]) + '.npy')

            curr_npy_path = get_mel_path(args.dir_mel, element)

            # print(curr_npy_path)
            # print(('/'.join(curr_npy_path.split('/')[:-1])))
            pathlib.Path('/'.join(curr_npy_path.split('/')[:-1]) + '/').mkdir(parents=True, exist_ok=True)
            np.save(curr_npy_path, y)
            print('saved :', curr_npy_path)
            filtered_list.append(element)

    q.put((type_feature, filtered_list, error_list))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='training script')
        parser.add_argument('--dir_wav', type=str, default='/media/bach3/dataset/kaka/music/fma_large',
                            help='path to dataset')
        parser.add_argument('--dir_mel', type=str, default='/media/iu/fma_large_mel/',
                            help='path to dataset')
        parser.add_argument('--num_part', type=int, default=8, help='num of cpu')

        args = parser.parse_args()

        ## parameters setting
        # args.frame_len_inMS = 50
        # args.frame_shift_inMS = 12.5
        # args.fmax = 7600
        # args.fmin = 125
        # args.n_fft = 1024
        # args.mel_dim = 80

        args.min_level_db = -100
        args.ref_level_db_mel = 20 + 0.2 * args.min_level_db  # adjust to fit in [0,1]
        args.ref_level_db_lin = 20
        # args.preemphasis_ratio = 0.97

        args.fft_size = 1024
        args.window = 1024
        args.hop = 512
        args.melBin = 128
        args.sample_rate = 22050


        print('Dataset to preprocess:', args.dir_wav)

        # track_id_to_tags = pickle.load(open(args.track_id_to_tags, 'rb'))

        track_ids_inst_and_genres_filtered = load_obj_curr_folder('../data_common/fma/track_ids_in_key_order')

        metalist = track_ids_inst_and_genres_filtered

        num_part = args.num_part
        part = int(len(metalist)/num_part)
        metalist_list = [metalist[part*i:part*(i+1)] for i in range(num_part-1)]
        metalist_list.append(metalist[part*(num_part-1):])

        print(len(metalist_list))

        # metalist_list_short = []
        # for i in range(len(metalist_list)):
        #     metalist_list_short.append(metalist_list[i][:5])
        # metalist_list = metalist_list_short

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

        with open('FMA_audio_filtered_track_file_path_list.p', 'wb') as handle:
            pickle.dump(filtered_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('FMA_audio_error_list.p', 'wb') as handle:
            pickle.dump(error_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for i in range(num_part):
            p_mels[i].join()

        print('error audio list : ', error_list)


    finally:
        for p in multiprocessing.active_children():
            p.terminate()
