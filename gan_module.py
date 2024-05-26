import tensorflow as tf
import numpy as np
import os
import pandas as pd
import datetime

import keras
from keras.layers import Dense, Dropout Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.layers import Input, Flatten, Multiply, Add, Reshape, UpSampling3D, BatchNormalization, LeakyReLU
from keras import Model
from keras.optimizers.legacy import Adam

from config import DataShape, DIR


class AdvAutoencoder():

    def __init__(self, latent_dim=256, data_shape=(DataShape.TIME_STEP, DataShape.ROW, DataShape.COL, DataShape.CHANNEL), optimizer=Adam(0.001, 0.5), encoder_weight=None, decoder_weight=None, missmatch=False):
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        # self.optimizer = optimizer

        # Build and config encoder & decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        if encoder_weight != None and decoder_weight != None:
            self.encoder.load_weights(encoder_weight, skip_mismatch=missmatch, by_name=missmatch)
            self.decoder.load_weights(decoder_weight, skip_mismatch=missmatch, by_name=missmatch)
            print("Encoder and Decoder MATCHED !")

        # Build and config discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build Autoencoder Adversarial
        data_input = Input(shape=(self.data_shape))
        encoded_repr = self.encoder(data_input)
        reconstructed_data = self.decoder(encoded_repr)
        data_validity = self.discriminator(encoded_repr)
        
        self.discriminator.trainable = False

        self.adv_autoencoder = Model(data_input, [reconstructed_data, data_validity])
        self.adv_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)
        tf.random.set_seed(40902)
        np.random.seed(40902)

        
        

    
    def build_encoder(self):
        initializer = tf.keras.initializers.HeNormal()
        data_input = Input(shape=self.data_shape)

        # Encoder 1
        encoder1 = Conv3D(
        filters=2,
        strides=(2,2,2),
        kernel_size=(3,3,3),
        padding="same",
        kernel_initializer=initializer
        )(data_input)
        encoder1 = LeakyReLU(0.2)(encoder1)
        encoder1 = Dropout(0.5)(encoder1)

        # Encoder 2
        encoder2 = Conv3D(
                filters=8,
                strides=(2,2,2),
                kernel_size=(3,3,3),
                padding="same",
                kernel_initializer=initializer
            )(encoder1)
        encoder2 = LeakyReLU(0.2)(encoder2)
        encoder2 = Dropout(0.5)(encoder2)


        # Encoder 3
        encoder3 = Conv3D(
                filters=16,
                strides=(2,2,2),
                kernel_size=(3,3,3),
                padding="same",
                kernel_initializer=initializer
            )(encoder2)
        encoder3 = LeakyReLU(0.2)(encoder3)
        encoder3 = Dropout(0.5)(encoder3)

        x =Flatten()(encoder3)
        mu = Dense(self.latent_dim)(x)
        log_var = Dense(self.latent_dim)(x)

        # Inisiasi noise
        sigma = tf.exp(0.5 * log_var)
        epsilon = tf.random.normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1]))

        latent_repr = Multiply()([sigma, epsilon])
        latent_repr = Add()([mu, latent_repr])


        return Model(data_input, latent_repr, name="encoder")
    
    
    def build_decoder(self):
        model = keras.Sequential(name='decoder')
        model.add(Dense(self.latent_dim, input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(2*8*32*self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((2,8,32,self.latent_dim)))
        model.add(ConvLSTM2D(32, (3,3), (1,1), return_sequences=True, recurrent_dropout=0.4, padding='same'))
        model.add(UpSampling3D((2,2,2)))
        model.add(ConvLSTM2D(32, (3,3), (1,1), return_sequences=True, recurrent_dropout=0.4,padding='same'))
        model.add(UpSampling3D((2,2,2)))
        model.add(ConvLSTM2D(32, (3,3), (1,1), return_sequences=True, recurrent_dropout=0.4,padding='same'))
        model.add(UpSampling3D((2,2,2)))
        model.add(Conv3DTranspose(2, (1,1,1), (1,1,1), padding='same'))
        return model



    def build_discriminator(self):
        model = keras.Sequential(name="discriminator")
        model.add(Dense(256, input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        return model
        
    def summary(self, layer_name =None):
        if layer_name != None:
            print(self.adv_autoencoder.get_layer(layer_name).summary())
        else:
            print(self.adv_autoencoder.summary(), flush=True)

    def train_model(self, dataset, epochs, batch_size=DataShape.BATCH_SIZE, save_every=10):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        valid = np.ones(shape=(batch_size, 1))
        invalid = np.zeros(shape=(batch_size,1))

        self.df = {
            "g_loss" : [],
            "g_accuracy" : [],
            "d_loss" : [],
            "d_accuracy" : [],
        }

        for epoch in range(1, epochs+1):
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            datas = dataset[idx]

            latent_fake = self.encoder(datas)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train Discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, invalid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # Train Generator
            g_loss = self.adv_autoencoder.train_on_batch(datas, [datas, valid])
            print(
                f"EPOCH-{epoch}: D_loss = {d_loss[0]}, D_acc = {d_loss[1]} , G_loss = {g_loss[0]} , G_acc = {g_loss[1]}", flush=True
            )
            self.df["g_loss"].append(g_loss[0])
            self.df["g_accuracy"].append(g_loss[1])
            self.df["d_loss"].append(d_loss[0])
            self.df["d_accuracy"].append(d_loss[1])

            if epoch % save_every == 0:
                self.adv_autoencoder.get_layer('encoder').save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"generator_encoder_weight_epoch_{epoch}_{current_time}.h5"))
                self.adv_autoencoder.get_layer('decoder').save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"generator_decoder_weight_epoch_{epoch}_{current_time}.h5"))
        
        df = pd.DataFrame(self.df)
        df.to_csv(os.path.join(DIR.MODEL_INFO, f"gan_adv_training_log_{current_time}.csv"))
        self.adv_autoencoder.get_layer('encoder').save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"generator_encoder_weight_epoch_{epochs}_{current_time}.h5"))
        self.adv_autoencoder.get_layer('decoder').save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"generator_dercoder_weight_epoch_{epochs}_{current_time}.h5"))


    def build_generator(self, encoder_weight=None, decoder_weight=None):
        if encoder_weight != None and decoder_weight != None:
            self.encoder.load_weights(encoder_weight)
            self.decoder.load_weights(decoder_weight)
        model = keras.Sequential()
        model.add(self.encoder)
        model.add(self.decoder)
        return model