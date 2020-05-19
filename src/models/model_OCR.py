from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adadelta, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.utils import residual_stack

K.set_learning_phase(1)
# from model_losses_OCR import ctc_lambda_func
import matplotlib.pyplot as plt
import pickle
import numpy as np
import itertools

global alphabet
global alphabet_num
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
alphabet_num = {}
i = 0
for tocken in alphabet:
    alphabet_num[str(tocken)] = i
    i += 1
alphabet_num['blanc'] = i


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class PlaqueOCR():
    def __init__(self, shape=(128, 64, 3), shapes=None, gru=None, weight=None, optimizers=None):
        self.OCR = self.BuildNN(shape=shape, gru=gru, weight=weight)

        self.to_train_OCR = self.BuildTrainNN(shape=shape, shapes=shapes, optimizers=optimizers)

    def BuildTrainNN(self, shape=None, shapes=None, optimizers=None):
        input_tensor = Input(name='train_input', shape=shape)
        y_pred = self.OCR(input_tensor)

        labels = Input(name='the_labels', shape=shapes, dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        adam = Adam(lr=0.0004)

        model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizers)

        return model

    def BuildNN(self, shape=None, gru=None, weight=None):
        img_w = shape[0]
        # Input Parameters
        img_h = shape[1]
        # Network parameters
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        time_dense_size = 32
        rnn_size = 512
        minibatch_size = 32
        unique_tokens = 26 + 10 + 1

        # Make Networkw
        inputs = Input(name='the_input', shape=shape, dtype='float32')  # (None, 128, 64, 1)

        # Convolution layer (VGG)
        inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
            inputs)  # (None, 128, 64, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

        inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
            inner)  # (None, 64, 32, 128)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

        inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

        inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
            inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

        inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
            inner)  # (None, 32, 4, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)

        # CNN to RNN
        inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
        inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

        # RNN layer
        gru_1 = GRU(gru, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 32, 512)
        gru_1b = GRU(gru, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)

        gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
        gru1_merged = BatchNormalization()(gru1_merged)

        gru_2 = GRU(gru, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(gru, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)
        reversed_gru_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_2b)

        gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
        gru2_merged = BatchNormalization()(gru2_merged)

        gru_3 = GRU(gru, return_sequences=True, kernel_initializer='he_normal', name='gru3')(gru2_merged)
        gru_3b = GRU(gru, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru3_b')(
            gru2_merged)
        reversed_gru_3b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_3b)

        gru3_merged = concatenate([gru_3, reversed_gru_3b])  # (None, 32, 1024)
        gru3_merged = BatchNormalization()(gru3_merged)

        # transforms RNN output to character activations:
        inner = Dense(unique_tokens, kernel_initializer='he_normal', name='dense2')(gru3_merged)  # (None, 32, 63)
        y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=inputs, outputs=y_pred)

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        if weight is not None:
            # load weights into new model
            model.load_weights(weight)
            print("Loaded model from disk")

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss='mse', optimizer=sgd)

        return model

    def train(self,
              x_train=None,
              y_train=None,
              batch_size=64,
              epochs=3,
              validation_data=None):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

        mc = ModelCheckpoint("OCR_latest.h5",
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             period=1)

        history = self.to_train_OCR.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_data=validation_data,
                                        callbacks=[es, mc])
        return history

    def train_generator(self,
                        gen_flow=None,
                        epochs=3,
                        steps_per_epoch=None,
                        validation_data=None):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

        mc = ModelCheckpoint("OCR_latest.h5",
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             period=1)

        history = self.to_train_OCR.fit_generator(gen_flow,
                                                  validation_data=validation_data,
                                                  steps_per_epoch=steps_per_epoch,
                                                  epochs=epochs,
                                                  callbacks=[es, mc])
        return history

    def predict(self, input=None, verbose=None):
        return self.OCR.predict(input, verbose=verbose)


class PlaqueOCR_res():
    def __init__(self, shape=(128, 64, 3), shapes=None, gru=None, weight=None, optimizers=None):
        self.OCR = self.BuildNN(shape=shape, gru=gru, weight=weight)

        self.to_train_OCR = self.BuildTrainNN(shape=shape, shapes=shapes, optimizers=optimizers)

    def BuildTrainNN(self, shape=None, shapes=None, weights=None, optimizers=None):
        input_tensor = Input(name='train_input', shape=shape)
        y_pred = self.OCR(input_tensor)

        labels = Input(name='the_labels', shape=shapes, dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizers)

        return model

    def BuildNN(self, shape=None, gru=None, weight=None):
        # Network parameters
        unique_tokens = 26 + 10 + 1

        # Make Networkw
        act = 'relu'
        input_data = Input(name='the_input', shape=shape, dtype='float32')  # (128, 64, 3)

        inner = residual_stack(filters=64, down_size=(2, 2))(input_data)  # (64, 32, 64)
        inner = residual_stack(filters=128, down_size=(2, 2))(inner)  # (32, 16, 128)
        # inner = residual_stack(filters=256, down_size=(0, 0))(inner)  # (32, 16, 256)
        # inner = residual_stack(filters=256, down_size=(0, 0))(inner)  # (32, 16, 256)
        inner = residual_stack(filters=512, down_size=(1, 2))(inner)  # (32, 8, 512)
        # inner = residual_stack(filters=512, down_size=(0, 0))(inner)  # (32, 8, 512)
        inner = residual_stack(filters=512, down_size=(1, 2))(inner)  # (32, 4, 512)

        # CNN to RNN
        inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
        inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

        # RNN layer
        gru_1 = GRU(gru, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 32, 512)
        gru_1b = GRU(gru, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            inner)
        reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_1b)

        gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
        gru1_merged = BatchNormalization()(gru1_merged)

        gru_2 = GRU(gru, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(gru, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)
        reversed_gru_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(gru_2b)

        gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
        gru2_merged = BatchNormalization()(gru2_merged)

        # transforms RNN output to character activations:
        inner = Dense(unique_tokens, kernel_initializer='he_normal', name='dense2')(gru2_merged)  # (None, 32, 63)
        y_pred = Activation('softmax', name='softmax')(inner)

        model = Model(inputs=input_data, outputs=y_pred)

        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        if weight is not None:
            # load weights into new model
            model.load_weights(weight)
            print("Loaded model from disk")

        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss='mse', optimizer=sgd)

        return model

    def train(self,
              x_train=None,
              y_train=None,
              batch_size=64,
              epochs=3,
              validation_data=None):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

        mc = ModelCheckpoint("OCR_latest.h5",
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             period=1)

        history = self.to_train_OCR.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_data=validation_data,
                                        callbacks=[es, mc])
        return history

    def train_generator(self,
                        gen_flow=None,
                        epochs=3,
                        steps_per_epoch=None,
                        validation_data=None):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

        mc = ModelCheckpoint("OCR_latest.h5",
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             period=1)

        history = self.to_train_OCR.fit_generator(gen_flow,
                                                  validation_data=validation_data,
                                                  steps_per_epoch=steps_per_epoch,
                                                  epochs=epochs,
                                                  callbacks=[es, mc])
        return history

    def predict(self, input=None, verbose=None):
        return self.OCR.predict(input, verbose=verbose)


def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def decode_batch(word_batch):
    out = word_batch
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


def decode_true(true):
    out = true
    ret = []
    for j in range(out.shape[0]):
        outstr = labels_to_text(out[j, :])
        ret.append(outstr)
    return ret
