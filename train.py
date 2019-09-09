import argparse

from keras.optimizers import SGD
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras import backend as K

from data_generator import *
from model import *


def main():
    parser = argparse.ArgumentParser(description='training script')

    # Experiment info
    parser.add_argument('--exp_info', type=str, default='', help='')
    parser.add_argument('--dataset', type=str, default='',
                        help='dataset to work on : msd / fma')

    # Path settings
    parser.add_argument('--dir_mel', type=str, default='', help='')

    # Weight loading from trained
    parser.add_argument('--load_weights', type=str, default=None, help='')

    # Data loading
    parser.add_argument('--num_negative_sampling', type=int, default=1, help='')

    parser.add_argument('--tag_vector_type', type=str, default='',
                        help='')

    parser.add_argument('--track_split', type=str, default='AB',
                        help='A+B / A / B ')

    # Name this track split. ex. TRS01
    parser.add_argument('--track_split_name', type=str, default='TRS01',
                        help='Track split name.')

    # Get tag split from pre-store file. ex. TGS01, TGS02
    parser.add_argument('--tag_split_name', type=str, default='TGS01',
                        help='Tag split name.')

    '''
        : ex. AB + TRS01 + TGS02  ->  track_keys_AB_TRS01_TGS02.p
              A + TRS02 + TGS02 ->  track_keys_A_TRS02_TGS02.p
    '''

    # Training
    parser.add_argument('--callback_degree', type=int, default=2, help='choose which callbacks to use')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=0.00000016, help='')
    parser.add_argument('--lrdecay', type=float, default=1e-6, help='')
    parser.add_argument('--workers', type=int, default=8, help='')
    parser.add_argument('--melBins', type=int, default=128, help='')
    parser.add_argument('--num_frame', type=int, default=130, help='')

    parser.add_argument('--don', type=bool, default=False, help='whether to use degree of negativity')
    parser.add_argument('--don_type', type=int, default=0, help='0:occur+cooccur / 1:cooccur')

    parser.add_argument('--hinge_margin', type=float, default=0.4, help='')
    parser.add_argument('--hm_neg_weight', type=float, default=1.0, help='')
    parser.add_argument('--hm_pos_weight', type=float, default=1.0, help='')
    parser.add_argument('--num_validation', type=int, default=1, help='')
    parser.add_argument('--save_per_n_epoch', type=int, default=2, help='')
    parser.add_argument('--epochs', type=int, default=0, help='')
    parser.add_argument('--num_random_data_aug', type=int, default=1, help='get more out of one audio')

    parser.add_argument('--loss_kind', type=int, default=0, help='0:hinge loss / 1:bpr loss')

    # parser.add_argument('--pretrained_audio_model', type=str, default='pretrained_audio_model/w-basic-ep_07-loss_0.14.h5', help='')
    parser.add_argument('--pretrained_audio_model', type=str, default=None, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')

    args = parser.parse_args()

    # Assign GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    K.tensorflow_backend._get_available_gpus()

    # Normalization (using precomputed values)
    if args.dataset == 'msd':
        args.global_mel_mean = 0.2262
        args.global_mel_std = 0.2579
    elif args.datset == 'fma':
        args.global_mel_mean = 0.2262
        args.global_mel_std = 0.2579

    data_common_path = os.path.join('data_common', args.dataset)
    data_tag_vector_path = os.path.join('data_tag_vector', args.dataset)

    # Load keys
    tag_split_file_name = data_common_path + '/tag_key_split_' + str(args.tag_split_name) + '.p'
    print('using tag split file : ', tag_split_file_name)

    args.train_tag_keys, args.test_tag_keys = pickle.load(open(tag_split_file_name, 'rb'))




    '''

    Experiment INFO

    exp_(exp info : anything)_(tag vector type)_(tag split name)_(track split name)_(track split)

    '''

    continued_epoch = 0

    if args.load_weights == None:
        exp_dir_info = 'exp_' + \
                       args.dataset + \
                       '_' + args.exp_info + \
                       '_' + args.tag_vector_type + \
                       '_' + str(args.tag_split_name) + \
                       '_' + str(args.track_split_name) + \
                       '_' + str(args.track_split)


    else:
        exp_dir_info = args.load_weights.split('/')[1]

        continued_epoch = int(args.load_weights.split('/')[-1].split('-')[2].split('_')[-1])

        args.dataset = args.load_weights.split('/')[1].split('_')[1]
        args.exp_info = args.load_weights.split('/')[1].split('_')[2]
        args.tag_vector_type = args.load_weights.split('/')[1].split('_')[3]
        args.tag_split_name = args.load_weights.split('/')[1].split('_')[4]
        args.track_split_name = args.load_weights.split('/')[1].split('_')[5]
        args.track_split = args.load_weights.split('/')[1].split('_')[6]
        print(' -> Continue training from epoch : ', continued_epoch)

    print('* dataset : ', args.dataset)
    print('* loaded exp : ', exp_dir_info)
    print('* tag_vector_type: ', args.tag_vector_type)
    print('* track_split : ', args.track_split)
    print('* track_split_name : ', args.track_split_name)
    print('* tag_split_name : ', args.tag_split_name)

    weight_name = './weights/' + exp_dir_info + '/w-ep_{epoch:02d}-loss_{val_loss:.2f}.h5'
    print('weight name will be :', weight_name)
    if not os.path.exists(os.path.dirname(weight_name)):
        os.makedirs(os.path.dirname(weight_name))

    csv_logger = CSVLogger('./weights/' + exp_dir_info + '/log.csv', append=True, separator='\n')

    '''
    track_ids_in_key_order
    tag_ids_in_key_order
    all_track_to_tag_bin_matrix
    track_key_to_id_dict 
    tag_key_to_id_dict 
    
    track_id_to_file_path_dict (if MSD)
    
    '''

    ## Load data
    args.track_key_to_tag_binary_matrix = pickle.load(open(data_common_path + '/all_track_to_tag_bin_matrix.p', 'rb'))
    args.tag_key_to_track_binary_matrix = args.track_key_to_tag_binary_matrix.T

    args.track_ids_in_key_order = pickle.load(open(data_common_path + '/track_ids_in_key_order.p', 'rb'))
    args.tag_ids_in_key_order = pickle.load(open(data_common_path + '/tag_ids_in_key_order.p', 'rb'))
    args.track_id_to_file_path_dict = pickle.load(open(data_common_path + '/track_id_to_file_path_dict.p', 'rb'))

    ## Tag Vector Selection
    if args.dataset == 'msd':
        args.dir_mel = '/media/irene/dataset/MSD/mel128'
        ## 1. GloVe word vector
        if args.tag_vector_type == 'glove':
            print('using GloVe word vector')
            args.tag_id_to_w2v_vector_dict = pickle.load(open(data_tag_vector_path + '/ttr_ont_tag_1126_to_glove_dict.p', 'rb'))
            args.tag_vec_dim = 300

        ## 2. Random
        elif args.tag_vector_type == 'random':
            print('using random vector')
            args.tag_id_to_w2v_vector_dict = pickle.load(open(data_tag_vector_path + '/ttr_ont_tag_1126_to_random_dict.p', 'rb'))
            args.tag_vec_dim = 300

        else:
            print('MSD dataset can have either glove or random as tag vector type')
            exit(0)

    if args.dataset == 'fma':
        args.dir_mel = '/media/iu/fma_large_mel'
        ## 1. pos / neg count 40-dim vector
        if args.tag_vector_type == 'inst-pncnt40':
            print('using instrument pos/neg 40 dim normalized count vector')
            args.tag_id_to_w2v_vector_dict = pickle.load(
                open(data_tag_vector_path + '/genre_id_to_inst_posneg40_cnt_norm_dict.p', 'rb'))
            args.tag_vec_dim = 40

        ## 2. pos / neg confidence 40-dim vector
        elif args.tag_vector_type == 'inst-pnconf40':
            print('using instrument pos/neg 40 dim normalized confidence vector')
            args.tag_id_to_w2v_vector_dict  = pickle.load(
                open(data_tag_vector_path + '/genre_id_to_inst_posneg40_conf_norm_dict.p', 'rb'))
            args.tag_vec_dim = 40

        ## 3. random vector
        elif args.tag_vector_type == 'random':
            print('using random vector')
            args.tag_id_to_w2v_vector_dict  = pickle.load(
                open(data_tag_vector_path + '/genre_id_to_random_vector_dict.p', 'rb'))
            args.tag_vec_dim = 40

        else:
            print('FMA dataset can have inst-pncnt40 / inst-pnconf40 / random as tag vector type')
            exit(0)

    # Build model
    args.data_common_path = data_common_path
    model = model_siamese_a2w_1fc(args)

    model.summary()
    if args.pretrained_audio_model is not None:
        print('loading pretrained model from ..', args.pretrained_audio_model)
        pre_model = model_audio_softmax(args)
        pre_model.load_weights(args.pretrained_audio_model)

        layer_names = ['am_input', 'am_conv_1', 'am_conv_2', 'am_conv_3', 'am_conv_4', 'am_conv_5']
        for layer_name in layer_names:
            curr_weight = pre_model.get_layer(layer_name).get_weights()
            model.get_layer(layer_name).set_weights(curr_weight)

    elif args.load_weights != None:
        print('loading weight.. :', args.load_weights)
        model.load_weights(args.load_weights)

    # Hinge loss
    def hinge_loss(_, y_pred):
        y_pos = y_pred[:, :1]
        y_neg = y_pred[:, 1:]
        loss = K.sum(K.maximum(0., args.hinge_margin - y_pos + y_neg))
        return loss

    # BPR loss : Works ONLY when num neg sample == 1
    def bpr_loss(_, y_pred):
        y_pos = y_pred[:, :1]
        y_neg = y_pred[:, 1:]
        loss = K.sum(1.0 - K.sigmoid(y_pos - y_neg))
        return loss

    sgd = SGD(lr=args.lr, decay=args.lrdecay, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=hinge_loss, metrics=['accuracy'])

    # Define data generators
    train_gen = DataGenerator(args, 'train')
    valid_gen = DataGenerator(args, 'valid')

    # Choosing callbacks
    if args.callback_degree == 0:
        callbacks = [ModelCheckpoint(monitor='val_loss', filepath=weight_name, verbose=0, save_best_only=False,
                                     mode='auto', period=args.save_per_n_epoch),
                     csv_logger]
    elif args.callback_degree == 1:
        callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='auto',
                                       min_lr=args.min_lr),
                     ModelCheckpoint(monitor='val_loss', filepath=weight_name, verbose=0, save_best_only=False, mode='auto',
                                     period=args.save_per_n_epoch),
                     csv_logger]
    elif args.callback_degree == 2:
        callbacks = [EarlyStopping(monitor='val_loss',patience=20,verbose=1,mode='auto'),
                     ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,verbose=1,mode='auto',min_lr=args.min_lr),
                     ModelCheckpoint(monitor='val_loss',filepath=weight_name,verbose=1,save_best_only=False,mode='auto',period=args.save_per_n_epoch),
                     csv_logger]

    # Training
    model.fit_generator(generator=train_gen,
                        workers=args.workers,
                        use_multiprocessing=True,
                        max_queue_size=args.workers,
                        epochs=args.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        initial_epoch = continued_epoch
                        )


    print('training done!')
    K.clear_session()


if __name__ == '__main__':
    main()

