from config import DataShape, DIR
from keras.layers import Input, Dropout, TimeDistributed, Flatten, Multiply, Add, Reshape, UpSampling3D
from keras.layers import Conv3D, ConvLSTM2D, LSTM, Dense
from keras import Model

import tensorflow as tf
import keras
import custom
import utils
import datetime
import os 
import pandas as pd


def vae_encoder(time_step=DataShape.TIME_STEP ,rows=DataShape.ROW,cols=DataShape.COL, channels=DataShape.CHANNEL):
    """
    Bagian encoder dari Variational Auto-encoder. 
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
            filters=channels,
            strides=(1,2,2),
            kernel_size=(3,3,3),
            padding="same",
            activation='elu'
        )(input_model)
    encoder = TimeDistributed(Dropout(0.5))(encoder)

    encoder = Conv3D(
            filters=8,
            strides=(1,2,2),
            kernel_size=(3,3,3),
            padding="same",
            activation='elu'
        )(encoder)
    
    encoder = TimeDistributed(Dropout(0.5))(encoder)

    encoder = Conv3D(
            filters=16,
            strides=(1,2,2),
            kernel_size=(3,3,3),
            padding="same",
            activation='elu'
        )(encoder)
    encoder = TimeDistributed(Dropout(0.5))(encoder)

    convlstm_1 = ConvLSTM2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        return_sequences=True,
        padding="same",
        recurrent_dropout=0.5
    )(encoder)

    out_1 = TimeDistributed(Dropout(0.5))(convlstm_1)

    ## Bottleneck ##
    x =TimeDistributed(Flatten())(out_1)
    bottleneck = LSTM(
        units=8*32,
        recurrent_dropout=0.5,
        return_sequences=True,
        activation='elu'
    )(x)
    mu = TimeDistributed(Dense(8*32))(bottleneck)
    log_var = TimeDistributed(Dense(8*32))(bottleneck)
    epsilon = tf.random.normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1], tf.shape(mu)[2]))
    # multi = tf.keras.layers.Multiply()([reshaped, out_1])
    sigma = tf.exp(0.5 * log_var)

    latent_space = Multiply()([sigma, epsilon])
    latent_space = Add()([mu, latent_space])

    
    

    model = Model(inputs=input_model, outputs=[mu, log_var, latent_space], name="encoder")
    return model

def vae_decoder(latent_output_shape):
    """
    Bagian decoder dari Variational Auto-encoder. 
    Digunakan untuk menerjemahkan representasi vektor terkompresi
    kedalam bentuk imitasi yang dibuat oleh model

    @Parameter
    latent_output_shape : Tuple
    """
    model_input= Input(shape=(latent_output_shape[1], latent_output_shape[2]))
    latent_space = TimeDistributed(Dense(8 * 32* 32))(model_input)
    reshaped = Reshape(target_shape=(DataShape.TIME_STEP, 8, 32, 32))(latent_space)

    convlstm_2 = ConvLSTM2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        return_sequences=True,
        recurrent_dropout=0.5,
        activation='elu'
    )(reshaped)

    x = UpSampling3D((1,2,2))(convlstm_2)

    convlstm_3 = ConvLSTM2D(
        filters=16,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        return_sequences=True,
        recurrent_dropout=0.5,
        activation='elu'
    )(x)

    x = UpSampling3D((1,2,2))(convlstm_3)

    convlstm_4 = ConvLSTM2D(
        filters=8,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        return_sequences=True,
        recurrent_dropout=0.5,
        activation='elu'
    )(x)

    x = UpSampling3D((1,2,2))(convlstm_4)

    convlstm_5 = ConvLSTM2D(
        filters=2,
        kernel_size=(3,3),
        strides=(1,1),
        padding="same",
        return_sequences=True,
        recurrent_dropout=0.5,
    )(x)

    model = Model(inputs=model_input, outputs=convlstm_5, name="decoder")
    return model

def vae(encoder_vae : keras.Model, decoder: keras.Model):
    """
    Digunakan untuk menggabungkan antara bagian encoder dan decoder pada
    Variational Auto-encoder

    @Parameter
    *encoder_vae : keras.Model
    *decoder : keras.Model
    name: string
    """
    mu, log_var, latent_space = encoder_vae(encoder_vae.input)
    reconstruct = decoder(latent_space)
    vae_model = Model(encoder_vae.input, reconstruct, name="vae")
    vae_model.add_loss(custom.kl_div_loss(mu, log_var))
    return vae_model

def train_vae(vae: keras.Model, dataset, epochs, optimizer=keras.optimizers.Adam(0.0001, 0.5), beta=0.05):
    """
    Digunakan untuk melatih model Variational Auto-encoder secara batch. 
    Proses training dilakukan secara custom agar dapat mengimplementasi
    KL-Div loss function beserta loss function lainnya.

    @Parameter
    *vae : keras.Model
    *dataset : numpy.array
    *epochs : int
    optimizer : keras.optimizers
    beta : float
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/train_vae_autoencoder/' + current_time + '/train'
    validation_log_dir = 'logs/train_vae_autoencoder/' + current_time + '/validation'
    mse_losses = []
    kl_losses = []
    total_losses = []
    val_losses = []


    train_data, val_data = utils.train_validation_split(dataset, 0.7)

    train_data= utils.create_generator(train_data)
    val_data = utils.create_generator(val_data)

    for epoch in range(1, epochs+1):            
        for batch_index in range(DataShape.BATCH_SIZE):
            data_gen,_ = next(train_data)
            val_gen,_ = next(val_data)
            with tf.GradientTape() as tape:
                y_pred = vae(data_gen)

                mse_loss = custom.reconstruction_loss(data_gen, y_pred)
                mse_losses.append(mse_loss.numpy())

                kl_loss = sum(vae.losses)
                kl_losses.append(kl_loss.numpy())


                loss = (beta/2) * kl_loss + mse_loss
                total_losses.append(loss.numpy())
                
                y_pred = vae(val_gen)
                val_losses.append(custom.reconstruction_loss(val_gen, y_pred).numpy()) 


                grads = tape.gradient(loss, vae.trainable_variables)
                optimizer.apply_gradients(zip(grads, vae.trainable_variables))

                print(f"EPOCH-{epoch}-{batch_index}: mse_loss : {mse_loss.numpy()}, kl_loss = {kl_loss.numpy()}, vae_loss = {loss.numpy()}", flush=True)

        if epoch % 10 == 0:
            y_pred = vae(val_gen)
            utils.plot_streamplot_data(y_pred[0], save_plot_to=os.path.join(DIR.SAVED_GENERATED_FILE, f"generated_epoch_{epoch+1}.gif"), show_figure=False)
    vae.save_weights(os.path.join(DIR.MODEL_WEIGHT_GENERATOR, f"beta_vae_lstm_{current_time}.h5"), overwrite=True)
    df = pd.DataFrame(
        {
            "mse_loss" : mse_losses,
            "kl_loss" : kl_losses,
            "total_loss" : total_losses,
            "val_loss" : val_losses
        }
    )
    df.to_csv(os.path.join(DIR.MODEL_INFO, f"loss_log_{current_time}.csv"))

