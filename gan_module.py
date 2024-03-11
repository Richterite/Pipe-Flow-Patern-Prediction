import tensorflow as tf
import numpy as np
import os
import pandas as pd
import datetime
from tqdm import tqdm

import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D
from keras.layers import TimeDistributed
from keras.layers import Input, Flatten
from keras import Model



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
    conv1 = TimeDistributed(Dropout(0.5))(conv1)

    conv2 = TimeDistributed(Conv2D(
        filters=64,
        kernel_size = (3,3),
        strides = (2,2),
        padding="same"
    ))(conv1)
    conv2 = TimeDistributed(Dropout(0.5))(conv2)
    flat = TimeDistributed(Flatten())(conv2)
    dense1 = TimeDistributed(Dense(128, activation="tanh"))(flat)
    out = TimeDistributed(Dense(1, activation='sigmoid'))(dense1)

    model = Model(inputs=input_model, outputs=out)
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

    dataset_generator = utils.create_generator(dataset)


    # Add model
    generator_model, discriminator_model = gan.layers
    generator_model.load_weights("./model_weight/generator/beta_vae_lstm_20231220-083027.h5")

    # Compile model
    discriminator_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
    gan.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])

    # List untuk menampung rerata Accuracy dan Loss
    generator_acc_avg = []
    generator_loss_avg = []
    discriminator_acc_avg = []
    discriminator_loss_avg = []

    print("Training in Progress...")

    for epoch in tqdm(range(epochs)):
        generator_acc = []
        generator_loss = []
        discriminator_acc = []
        discriminator_loss = []
        for index in range(DataShape.BATCH_SIZE):
            data_array, model_base = next(dataset_generator)

            # Phase 1: Train Discriminator
            # Create fake image
            noise = tf.random.normal(shape=[1,DataShape.TIME_STEP,DataShape.ROW,DataShape.COL, DataShape.CHANNEL])
            fake_data = generator_model.predict(noise, verbose=0)
    

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
            hist_generator = gan.train_on_batch(data_array, y, return_dict=True)
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
            generator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"generator_weight_epoch_{epoch+1}.h5"), overwrite=True)
            discriminator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_DISCRIMINATOR, f"discriminator_weight_epoch_{epoch+1}.h5"), overwrite=True)

    # Save semua pada akhir training
    generator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, "generator_weight_epoch_END.h5"), overwrite=True)
    discriminator_model.save_weights(os.path.join(DIR.MODEL_WEIGHT_DISCRIMINATOR, "discriminator_weight_epoch_END.h5"), overwrite=True)
    df = pd.DataFrame(
        {
            "Discriminator_Acc" : discriminator_acc_avg,
            "Discriminator_loss": discriminator_loss_avg,
            "Generator_Acc" : generator_acc_avg,
            "Generator_Loss": generator_loss_avg
        }
    )
    df.to_csv(os.path.join(DIR.MODEL_INFO, f"/gan_training_log_{current_time}.csv"))


