from __future__ import print_function

from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Flatten, concatenate, Embedding
from keras.models import Model
from keras.regularizers import l2


def model_siamese_a2w_1fc(args):

    num_frame = args.num_frame
    num_negative_sampling = args.num_negative_sampling

    # audio anchor
    model_siamese_a2w_1fc.audio_input = Input(shape=(num_frame, 128), name='am_input')

    # positive word
    model_siamese_a2w_1fc.pos_item = Input(shape=(args.tag_vec_dim,))

    # negative word
    model_siamese_a2w_1fc.neg_items = [Input(shape=(args.tag_vec_dim,)) for j in range(num_negative_sampling)]

    # audio model
    conv1 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_1')
    activ1 = Activation('relu')
    MP1 = MaxPool1D(pool_size=4)
    conv2 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_2')
    activ2 = Activation('relu')
    MP2 = MaxPool1D(pool_size=4)
    conv3 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_3')
    activ3 = Activation('relu')
    MP3 = MaxPool1D(pool_size=4)
    conv4 = Conv1D(128, 2, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_4')
    activ4 = Activation('relu')
    MP4 = MaxPool1D(pool_size=2)
    conv5 = Conv1D(100, 1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_5')

    # (batch, steps, features) -> (batch, features)
    GP = GlobalAvgPool1D()

    # tag model
    fc1 = Dense(100)

    # Audio Anchor
    anchor_conv1 = conv1(model_siamese_a2w_1fc.audio_input)
    anchor_activ1 = activ1(anchor_conv1)
    anchor_MP1 = MP1(anchor_activ1)
    anchor_conv2 = conv2(anchor_MP1)
    anchor_activ2 = activ2(anchor_conv2)
    anchor_MP2 = MP2(anchor_activ2)
    anchor_conv3 = conv3(anchor_MP2)
    anchor_activ3 = activ3(anchor_conv3)
    anchor_MP3 = MP3(anchor_activ3)
    anchor_conv4 = conv4(anchor_MP3)
    anchor_activ4 = activ4(anchor_conv4)
    anchor_MP4 = MP4(anchor_activ4)
    anchor_conv5 = conv5(anchor_MP4)
    model_siamese_a2w_1fc.anchor_output = GP(anchor_conv5)

    # POS word item
    model_siamese_a2w_1fc.pos_item_output = fc1(model_siamese_a2w_1fc.pos_item)

    # NEG word item
    model_siamese_a2w_1fc.neg_item_output = [fc1(neg_item) for neg_item in model_siamese_a2w_1fc.neg_items]


    RQD_p = dot([model_siamese_a2w_1fc.anchor_output, model_siamese_a2w_1fc.pos_item_output], axes=1, normalize=True)
    RQD_ns = [dot([model_siamese_a2w_1fc.anchor_output, neg_item], axes=1, normalize=True) for neg_item
              in model_siamese_a2w_1fc.neg_item_output]

    prob = concatenate([RQD_p] + RQD_ns)

    # for now softmax & categorical cross entropy loss
    output = Activation('linear')(prob)

    model = Model(inputs=[model_siamese_a2w_1fc.audio_input, model_siamese_a2w_1fc.pos_item] +
                         model_siamese_a2w_1fc.neg_items, outputs=output)

    return model




def model_audio_softmax(args):
    num_frame = args.num_frame

    # audio input
    model_audio_softmax.audio_input = Input(shape=(num_frame, 128), name='am_input')

    # item model **audio**
    conv1 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_1')
    activ1 = Activation('relu')
    MP1 = MaxPool1D(pool_size=4)
    conv2 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_2')
    activ2 = Activation('relu')
    MP2 = MaxPool1D(pool_size=4)
    conv3 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_3')
    activ3 = Activation('relu')
    MP3 = MaxPool1D(pool_size=4)
    conv4 = Conv1D(128, 2, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_4')
    activ4 = Activation('relu')
    MP4 = MaxPool1D(pool_size=2)
    conv5 = Conv1D(100, 1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform', name='am_conv_5')

    # (batch, steps, features) -> (batch, features)
    GP = GlobalAvgPool1D()

    fc_f = Dense(50, input_shape=(100,))

    # pos anchor
    audio_conv1 = conv1(model_audio_softmax.audio_input)
    audio_activ1 = activ1(audio_conv1)
    audio_MP1 = MP1(audio_activ1)
    audio_conv2 = conv2(audio_MP1)
    audio_activ2 = activ2(audio_conv2)
    audio_MP2 = MP2(audio_activ2)
    audio_conv3 = conv3(audio_MP2)
    audio_activ3 = activ3(audio_conv3)
    audio_MP3 = MP3(audio_activ3)
    audio_conv4 = conv4(audio_MP3)
    audio_activ4 = activ4(audio_conv4)
    audio_MP4 = MP4(audio_activ4)
    audio_conv5 = conv5(audio_MP4)
    model_audio_softmax.audio_sem = GP(audio_conv5)
    model_audio_softmax.audio_fc = fc_f(model_audio_softmax.audio_sem)
    model_audio_softmax.output = Activation('sigmoid')(model_audio_softmax.audio_fc)

    model = Model(inputs=model_audio_softmax.audio_input, outputs=model_audio_softmax.output)

    return model

