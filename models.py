# Defines the models used in the experiments

import numpy as np
from keras.layers import Dense, Input, merge, Activation, Dropout, Flatten
from keras.models import Model
from keras.layers.convolutional import AtrousConvolution1D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils.np_utils import conv_output_length

from util import one_hot, NUM_STYLES
from music import NUM_CLASSES, NOTES_PER_BAR, NUM_KEYS
from keras.models import load_model

class CausalAtrousConvolution1D(AtrousConvolution1D):
    def __init__(self, nb_filter, filter_length, init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample_length=1, atrous_rate=1, W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, causal=False, **kwargs):
        super(CausalAtrousConvolution1D, self).__init__(nb_filter, filter_length, init, activation, weights,
                                                        border_mode, subsample_length, atrous_rate, W_regularizer,
                                                        b_regularizer, activity_regularizer, W_constraint, b_constraint,
                                                        bias, **kwargs)
        self.causal = causal
        if self.causal and border_mode != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def get_output_shape_for(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.atrous_rate * (self.filter_length - 1)

        length = conv_output_length(input_length,
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0],
                                    dilation=self.atrous_rate)

        return (input_shape[0], length, self.nb_filter)

    def call(self, x, mask=None):
        if self.causal:
            x = K.asymmetric_temporal_padding(x, self.atrous_rate * (self.filter_length - 1), 0)
        return super(CausalAtrousConvolution1D, self).call(x, mask)

def residual_block(x):
    original_x = x
    # TODO: initalization, regularization?
    # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
    tanh_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                         bias=use_bias,
                                         name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
                                         W_regularizer=l2(res_l2))(x)
    sigm_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                         bias=use_bias,
                                         name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid',
                                         W_regularizer=l2(res_l2))(x)
    x = layers.Merge(mode='mul', name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

    res_x = layers.Convolution1D(nb_filters, 1, border_mode='same', bias=use_bias,
                                 W_regularizer=l2(res_l2))(x)
    skip_x = layers.Convolution1D(nb_filters, 1, border_mode='same', bias=use_bias,
                                  W_regularizer=l2(res_l2))(x)
    res_x = layers.Merge(mode='sum')([original_x, res_x])
    return res_x, skip_x

def pre_model(time_steps):
    # Primary input
    note_input = Input(shape=(time_steps, NUM_CLASSES), name='note_input')

    # Context inputs
    # beat_input = Input(shape=(time_steps, NOTES_PER_BAR), name='beat_input')
    beat_input = Input(shape=(time_steps, 2), name='beat_input')
    completion_input = Input(shape=(time_steps, 1), name='completion_input')
    style_input = Input(shape=(time_steps, NUM_STYLES), name='style_input')
    context = merge([completion_input, beat_input, style_input], mode='concat')

    # Build network stack
    x = note_input

    # Create a distributerd representation of context
    context = GRU(128, return_sequences=True)(context)
    context = BatchNormalization()(context)
    context = Activation('relu')(context)

    """
    skip_connections = []

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out, skip_out = residual_block(out)
            skip_connections.append(skip_out)
    """

    # Simple convs
    for i, nb_filters in enumerate([32, 64, 128, 256]):
        x = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, causal=True)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = merge([x, context], mode='concat')
    x = GRU(nb_filters)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return [note_input, beat_input, completion_input, style_input], x


def note_model(time_steps):
    inputs, x = pre_model(time_steps, False)

    # Multi-label
    policy = Dense(NUM_CLASSES, name='policy', activation='softmax')(x)
    value = Dense(1, name='value', activation='linear')(x)

    model = Model(inputs, [policy, value])
    #model.load_weights('data/supervised.h5', by_name=True)
    # Create value output
    return model


def supervised_model(time_steps):
    inputs, x = pre_model(time_steps)

    # Multi-label
    x = Dense(NUM_CLASSES, name='policy', activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def note_preprocess(env, x):
    note, beat = x
    return (one_hot(note, NUM_CLASSES), one_hot(beat, NOTES_PER_BAR))
