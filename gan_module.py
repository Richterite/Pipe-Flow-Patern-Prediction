import tensorflow as tf
import numpy as np
import os
import pandas as pd
import datetime
from tqdm import tqdm

import keras
import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.layers import TimeDistributed, LeakyReLU
from keras.layers import Input, Flatten, Multiply, Add, Reshape, UpSampling3D, BatchNormalization
from keras import Model
from keras.optimizers.legacy import Adam

import custom
import utils
from config import DataShape, DIR




def discriminator(time_step=DataShape.TIME_STEP, row=DataShape.ROW, col=DataShape.COL, channel=DataShape.CHANNEL):
    """
    Digunakan untuk membuat model discriminator GAN
    
    @Parameter
    time_step : int
    row : int
    col : int
    channel : int
    """
    input_model = Input(shape=(time_step, row, col, channel))
    conv1 = TimeDistributed(Conv2D(
        filters=32,
        kernel_size = (3,3),
        strides = (1,1),
        padding="same"
    ))(input_model)
    conv1 =LeakyReLU(alpha=0.05)(conv1)

    conv2 = TimeDistributed(Conv2D(
        filters=64,
        kernel_size = (3,3),
        strides = (2,2),
        padding="same"
    ))(conv1)
    conv2 =LeakyReLU(alpha=0.05)(conv2)
    flat = TimeDistributed(Flatten())(conv2)
    dense1 = TimeDistributed(Dense(512))(flat)
    dense1 = TimeDistributed(Dense(256))(flat)
    dense1 = TimeDistributed(Dense(128))(flat)
    out = TimeDistributed(Dense(1, activation='sigmoid'))(dense1)

    model = Model(inputs=input_model, outputs=out)
    return model



def generator_encoder(time_step=DataShape.TIME_STEP ,rows=DataShape.ROW,cols=DataShape.COL, channels=DataShape.CHANNEL):
    """
    Bagian encoder dari encoder generator.
    Digunakan dalam proses encoding data menjadi
    representasi vektor terkompresi dalam latent space

    @Parameter
    time_step : int
    rows : int
    cols : int
    channels : int
    """
    input_model = Input(shape=(time_step,rows,cols, channels))
    encoder = Conv3D(
            filters=1,
            strides=(1,2,2),
            kernel_size=(3,3,3),
            padding="same",
            activation='elu'
        )(input_model)
    encoder = LeakyReLU(alpha=0.05)(encoder)

    encoder = Conv3D(
            filters=8,
            strides=(1,2,2),
            kernel_size=(3,3,3),
            padding="same",
            activation='elu'
        )(encoder)

    encoder = LeakyReLU(alpha=0.05)(encoder)

    encoder = Conv3D(
            filters=16,
            strides=(1,2,2),
            kernel_size=(3,3,3),
            padding="same",
            activation='elu'
        )(encoder)
    encoder = LeakyReLU(alpha=0.05)(encoder)

    convlstm_1 = ConvLSTM2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        return_sequences=True,
        padding="same",
        recurrent_dropout=0.2
    )(encoder)

    out_1 = LeakyReLU(alpha=0.05)(convlstm_1)

    ## Bottleneck ##
    x =TimeDistributed(Flatten())(out_1)
    latent_space =  TimeDistributed(Dense( 256 , name="latent_space"))(x)

    model = Model(inputs=input_model, outputs=latent_space, name="generator_encoder")
    return model

def set_trinability_model(model, trainable):
    """
    Digunakan untuk mengaktif/non-aktifkan 
    layer model selama proses training berlangsung.

    @Parameter
    *model: keras.Model
    *trainable : boolean
    """
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def gan_model(generator_obj, discriminator_obj):
    """
    Digunakan untuk menggabungkan antara 
    generator dengan discrimintor untuk
    membentuk model GAN

    @Parameter
    *generator_obj : keras.Model
    *discriminator_obj : keras.Model
    """
    model = Sequential()
    model.add(generator_obj)
    set_trinability_model(discriminator_obj, False)
    model.add(discriminator_obj)
    return model



def train_gan(gan: keras.Model, dataset, batch_size=DataShape.BATCH_SIZE, epochs=100):
    """
    Digunakan untuk proses training model GAN.
    Model gan harus terdiri atas generator dan discrimintor

    @Parameter
    *gan : keras.Model
    *dataset: numpy.array
    batch_size: int
    epochs: int
    """
    # Dataset disini pakai yang sudah dipisah per batch saja dan stacked=True

    # Summary Writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    generator_log_dir = 'logs/train_gan/' + current_time + '/generator'
    discriminator_log_dir = 'logs/train_gan/' + current_time + '/discriminator'

    generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
    discriminator_summary_writer = tf.summary.create_file_writer(discriminator_log_dir)


    ## NOTE: Perhatikan apa yang akan dibikin custom, dan lihat utils.create_pressure_matrix
    dataset_generator = utils.create_generator(*dataset)


    # Add model
    generator_model, discriminator_model = gan.layers
    # generator_model.load_weights("./model_weight/generator/beta_vae_lstm_20231220-083027.h5")

    # Compile model
    discriminator_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
    gan.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])

    # List untuk menampung rerata Accuracy dan Loss
    generator_acc_avg = []
    generator_loss_avg = []
    discriminator_acc_avg = []
    discriminator_loss_avg = []

    print("Training in Progress...")

    for epoch in tqdm(range(1, epochs+1)):
        generator_acc = []
        generator_loss = []
        discriminator_acc = []
        discriminator_loss = []
        for index in range(batch_size):
            data_array, model_base = next(dataset_generator)
            # print(f"data asli : {data_array.shape}" ,flush=True)
            # print(f"data model base : {model_base.shape}" ,flush=True)

            # Phase 1: Train Discriminator
            # Create fake image
            # noise = tf.random.normal(shape=[1,DataShape.TIME_STEP,DataShape.ROW,DataShape.COL, DataShape.CHANNEL])
            fake_data = generator_model.predict(model_base, verbose=0)
    

            x = np.concatenate([fake_data, data_array])
            y = np.concatenate([np.zeros(shape=(1, DataShape.TIME_STEP, 1), dtype=np.int32), 
                                             np.ones(shape=(1, DataShape.TIME_STEP, 1), dtype=np.int32)])
            
         
            set_trinability_model(discriminator_model, True)
            hist_dicriminator = discriminator_model.train_on_batch(x, y, return_dict=True)
            discriminator_loss.append(hist_dicriminator['loss'])
            discriminator_acc.append(hist_dicriminator['acc'])

            # Phase 2: Train generator
            y = np.ones(shape=(1, DataShape.TIME_STEP, 1), dtype=np.int32)
            set_trinability_model(discriminator_model, False)
            hist_generator = gan.train_on_batch(model_base, y, return_dict=True)
            generator_loss.append(hist_generator['loss'])
            generator_acc.append(hist_generator['acc'])
            print(f'itter-{index}: g_loss: {generator_loss[len(generator_loss) - 1]} d_loss: {discriminator_loss[len(discriminator_loss) - 1]}\n',
                  flush=True)

        g_loss_average = np.average(generator_loss)
        generator_loss_avg.append(g_loss_average)

        d_loss_average = np.average(discriminator_loss)
        discriminator_loss_avg.append(d_loss_average)

        g_acc_average = np.average(generator_acc)
        generator_acc_avg.append(g_acc_average)

        d_acc_average = np.average(discriminator_acc)
        discriminator_acc_avg.append(d_acc_average)


        with generator_summary_writer.as_default():
            tf.summary.scalar('accuracy',g_acc_average, step=epoch)
            tf.summary.scalar('loss', g_loss_average, step=epoch)

        
        with discriminator_summary_writer.as_default():
            tf.summary.scalar('accuracy', d_acc_average, step=epoch)
            tf.summary.scalar('loss', d_loss_average, step=epoch)

        
        print(f"EPOCH-{epoch+1} : g_loss_avg: {g_loss_average} d_loss_avg: {d_loss_average}", flush=True)
  
        
        # Save weight per 10 epoch
        if epoch % 10 == 0:
            generator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"generator_weight_epoch_{epoch+1}_{current_time}.h5"), overwrite=True)
            discriminator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_DISCRIMINATOR, f"discriminator_weight_epoch_{epoch+1}_{current_time}.h5"), overwrite=True)

    # Save semua pada akhir training
    generator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"generator_weight_epoch_END_{current_time}.h5"), overwrite=True)
    discriminator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_DISCRIMINATOR, f"discriminator_weight_epoch_END_{current_time}.h5"), overwrite=True)
    df = pd.DataFrame(
        {
            "Discriminator_Acc" : discriminator_acc_avg,
            "Discriminator_loss": discriminator_loss_avg,
            "Generator_Acc" : generator_acc_avg,
            "Generator_Loss": generator_loss_avg,
        }
    )
    df.to_csv(os.path.join(DIR.MODEL_INFO, f"gan_training_log_{current_time}.csv"))




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