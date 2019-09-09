# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import os, argparse, pickle, multiprocessing
from multiprocessing import Process, Queue
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def save_obj_curr_folder(obj, name):
    with open(name + '.p', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_curr_folder(name):
    with open(name + '.p', 'rb') as f:
        return pickle.load(f)


def extract_audio_embed(args,
                        p_idx,
                        train_tag_keys,
                        test_tag_keys,
                        curr_track_ids_in_order,
                        track_id_to_key_dict,
                        track_id_to_file_path_dict,
                        tag_ids_in_key_order,
                        q):

    from keras.models import Model
    from model import model_siamese_a2w_1fc
    import numpy as np

    tag_id_to_w2v_vector_dict = args.tag_id_to_w2v_vector_dict

    # Build model
    model = model_siamese_a2w_1fc(args)

    model.load_weights(args.load_weights)


    audio_model = Model(inputs=model_siamese_a2w_1fc.audio_input,
                        outputs=model_siamese_a2w_1fc.anchor_output)
    word_model = Model(inputs=model_siamese_a2w_1fc.pos_item,
                       outputs=model_siamese_a2w_1fc.pos_item_output)

    if p_idx == 0:
        # (1) extract tag embeddings
        tag_key_to_embedding_dict = dict()
        all_tag_keys = train_tag_keys + test_tag_keys

        for key in all_tag_keys:
            curr_tag_word_vec = np.array(tag_id_to_w2v_vector_dict[tag_ids_in_key_order[key]]).reshape(1, -1)
            predicted = word_model.predict(curr_tag_word_vec)
            tag_key_to_embedding_dict[key] = predicted

    # (2) extract audio embeddings
    track_key_to_embedding_dict = dict()

    for t_id in tqdm(curr_track_ids_in_order):
        key = track_id_to_key_dict[t_id]

        # load anchor audio
        mel_path = track_id_to_file_path_dict[t_id]
        curr_audio = np.load(os.path.join(args.dir_mel, mel_path.replace('.mp3', '.npy')))

        while curr_audio.shape[0] < args.num_frame:
            curr_audio = np.concatenate((curr_audio, curr_audio), axis=0)

        num_seg = int(curr_audio.shape[0] / args.num_frame)
        curr_audio_segs = []
        for seg_idx in range(num_seg):
            curr_audio_seg = curr_audio[seg_idx * args.num_frame: (seg_idx + 1) * args.num_frame]
            curr_audio_seg -= args.global_mel_mean
            curr_audio_seg /= args.global_mel_std
            curr_audio_segs.append(curr_audio_seg)

        predicted = audio_model.predict(np.array(curr_audio_segs))
        track_key_to_embedding_dict[key] = np.mean(predicted, axis=0)
    if p_idx == 0:
        q.put((tag_key_to_embedding_dict, track_key_to_embedding_dict))
    else:
        tag_key_to_embedding_dict = dict()
        q.put((tag_key_to_embedding_dict, track_key_to_embedding_dict))



def main():
    try:
        parser = argparse.ArgumentParser(description='training script')

        # Weight loading from trained
        parser.add_argument('--load_weights', type=str, default='')

        # Path settings
        parser.add_argument('--dir_mel', type=str, default='', help='')

        # Data loading
        parser.add_argument('--num_part', type=int, default=12, help='num of cpu')

        parser.add_argument('--num_negative_sampling', type=int, default=1, help='')

        args = parser.parse_args()

        args.global_mel_mean = 0.2262
        args.global_mel_std = 0.2579
        args.num_frame = 130


        if args.load_weights is not None:
            exp_dir_info = args.load_weights.split('/')[1]

            args.dataset = args.load_weights.split('/')[1].split('_')[1]
            args.exp_info = args.load_weights.split('/')[1].split('_')[2]
            args.tag_vector_type = args.load_weights.split('/')[1].split('_')[3]
            args.tag_split_name = args.load_weights.split('/')[1].split('_')[4]
            args.track_split_name = args.load_weights.split('/')[1].split('_')[5]
            args.track_split = args.load_weights.split('/')[1].split('_')[6]

            print('* dataset : ', args.dataset)
            print('* loaded exp : ', exp_dir_info)
            print('* tag_vector_type: ', args.tag_vector_type)
            print('* track_split : ', args.track_split)
            print('* track_split_name : ', args.track_split_name)
            print('* tag_split_name : ', args.tag_split_name)

        else:
            print('Need weight path to load.')
            exit(0)

        data_common_path = os.path.join('data_common', args.dataset)
        data_tag_vector_path = os.path.join('data_tag_vector', args.dataset)
        args.data_common_path = data_common_path

        # Load data

        args.tag_key_to_track_binary_matrix = pickle.load(
            open(args.data_common_path + '/all_tag_to_track_bin_matrix.p', 'rb'))

        args.track_key_to_tag_binary_matrix = args.tag_key_to_track_binary_matrix.T

        # Tag Vector Selection
        if args.dataset == 'msd':
            args.dir_mel = '/media/irene/dataset/MSD/mel128'

            # 1. GloVe word vector
            if args.tag_vector_type == 'glove':
                print('using GloVe word vector')
                args.tag_id_to_w2v_vector_dict = pickle.load(
                    open(data_tag_vector_path + '/ttr_ont_tag_1126_to_glove_dict.p', 'rb'))
                args.tag_vec_dim = 300

            # 2. Random
            elif args.tag_vector_type == 'random':
                print('using random vector')
                args.tag_id_to_w2v_vector_dict = pickle.load(
                    open(data_tag_vector_path + '/ttr_ont_tag_1126_to_random_dict.p', 'rb'))
                args.tag_vec_dim = 300

            else:
                print('MSD dataset can have either glove or random as tag vector type')
                exit(0)

        if args.dataset == 'fma':
            args.dir_mel = '/media/iu/fma_large_mel'

            # 1. pos / neg count 40-dim vector
            if args.tag_vector_type == 'inst-pncnt40':
                print('using instrument pos/neg 40 dim normalized count vector')
                args.tag_id_to_w2v_vector_dict = pickle.load(
                    open(data_tag_vector_path + '/genre_id_to_inst_posneg40_cnt_norm_dict.p', 'rb'))
                args.tag_vec_dim = 40

            # 2. pos / neg confidence 40-dim vector
            elif args.tag_vector_type == 'inst-pnconf40':
                print('using instrument pos/neg 40 dim normalized confidence vector')
                args.tag_id_to_w2v_vector_dict = pickle.load(
                    open(data_tag_vector_path + '/genre_id_to_inst_posneg40_conf_norm_dict.p', 'rb'))
                args.tag_vec_dim = 40

            # 3. random vector
            elif args.tag_vector_type == 'random':
                print('using random vector')
                args.tag_id_to_w2v_vector_dict = pickle.load(
                    open(data_tag_vector_path + '/genre_id_to_random_vector_dict.p', 'rb'))
                args.tag_vec_dim = 40

            else:
                print('FMA dataset can have inst-pncnt40 / inst-pnconf40 / random as tag vector type')
                exit(0)

        # Load keys
        tag_split_file_name = data_common_path + '/tag_key_split_' + str(args.tag_split_name) + '.p'
        train_tag_keys, test_tag_keys = pickle.load(open(tag_split_file_name, 'rb'))

        track_ids_in_key_order = pickle.load(open(data_common_path + '/track_ids_in_key_order.p', 'rb'))
        tag_ids_in_key_order = pickle.load(open(data_common_path + '/tag_ids_in_key_order.p', 'rb'))
        track_id_to_key_dict = pickle.load(open(data_common_path + '/track_id_to_key_dict.p', 'rb'))
        track_id_to_file_path_dict = pickle.load(open(data_common_path + '/track_id_to_file_path_dict.p', 'rb'))

        args.min_level_db = -100
        args.ref_level_db_mel = 20 + 0.2 * args.min_level_db  # adjust to fit in [0,1]
        args.ref_level_db_lin = 20

        args.fft_size = 1024
        args.window = 1024
        args.hop = 512
        args.melBin = 128
        args.sample_rate = 22050

        # Embeddings will be saved in a dict
        track_key_to_embedding_dict_list = []
        tag_key_to_embedding_dict_list = []

        num_part = args.num_part
        part = int(len(track_ids_in_key_order) / num_part)
        tracklist_list = [track_ids_in_key_order[part * i:part * (i + 1)] for i in range(num_part - 1)]
        tracklist_list.append(track_ids_in_key_order[part * (num_part - 1):])

        q = Queue()
        p_mels = []
        for i in range(num_part):
            p_mels.append(Process(target=extract_audio_embed, args=(args,
                                                                    i,
                                                                    train_tag_keys,
                                                                    test_tag_keys,
                                                                    tracklist_list[i],
                                                                    track_id_to_key_dict,
                                                                    track_id_to_file_path_dict,
                                                                    tag_ids_in_key_order,
                                                                    q)))

        for i in range(num_part):
            p_mels[i].daemon = True
            p_mels[i].start()

        for i in range(num_part):
            tag_key_to_embedding_dict, track_key_to_embedding_dict = q.get()
            track_key_to_embedding_dict_list.append(track_key_to_embedding_dict)
            tag_key_to_embedding_dict_list.append(tag_key_to_embedding_dict)

        for i in range(num_part):
            p_mels[i].join()


        all_track_key_to_embedding_dict = dict()
        for i in range(num_part):
            for k, v in track_key_to_embedding_dict_list[i].items():
                all_track_key_to_embedding_dict[k] = v

        all_tag_key_to_embedding_dict = dict()
        for i in range(num_part):
            for k, v in tag_key_to_embedding_dict_list[i].items():
                all_tag_key_to_embedding_dict[k] = v

        # (3) saving audio / word embeddings
        print('saving')
        weight_name = args.load_weights.split('/')[-1]

        save_path = 'pred_embeddings/' + exp_dir_info + '/'
        print(os.path.dirname(save_path))

        if not os.path.exists(os.path.dirname(save_path)):
            print('os mkdir')
            os.makedirs(os.path.dirname(save_path))

        with open(os.path.join(save_path, 'track_key_to_embedding_dict_' + weight_name + '.p'), 'wb') as handle:
            pickle.dump(all_track_key_to_embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_path, 'tag_key_to_embedding_dict_' + weight_name + '.p'), 'wb') as handle:
            pickle.dump(all_tag_key_to_embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    finally:
        for p in multiprocessing.active_children():
            p.terminate()



if __name__ == '__main__':
    main()
    print('done.')
