import numpy as np
import tensorflow as tf
import keras
import gan_module
from config import DataShape, DIR
import os
import vae_module
import keras




if __name__ == "__main__":
    # Cek apakah struktur folder sudah tepat
    # Jika folder tidak ada maka buat folder
    directory = dir(DIR)[:5]
    for target in directory:
        folder = getattr(DIR, target)
        if not os.path.isdir(folder):
            os.makedirs(folder)
    # Uncomment Jika dibutuhkan
    # datagen = NumericalDataGenerator()
    # datagen.generateData(plot_data=True)


    # Train GANN
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    del gpus

    x_data = np.load("./data/data_step16/x_data_16step.npy")
    y_data = np.load("./data//data_step16/y_data_16step.npy")
 
    # height_array = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44]


    datagen = utils.batch_time_series_dataset([x_data, y_data], time_step=DataShape.TIME_STEP, batch_size=DataShape.BATCH_SIZE ,stack_data=True)

    # VAE PRE-TRAIN
    # encoder = vae_module.vae_encoder()
    # # print(encoder.summary())
    # decoder = vae_module.vae_decoder(encoder.layers[-1].output_shape)
    # # print(decoder.summary())
    # vae_model = vae_module.vae(encoder, decoder)
    # vae_model.load_weights('./model_weight/generator/beta_vae_lstm_20231220-083027.h5')

    # y_pred = vae_model(np.expand_dims(datagen[30], axis=0))
    # print(y_pred.shape)
    # np.save("hasil_beta_vae.npy", y_pred)
    # print(vae_model.summary(), flush=True)
    # vae_module.train_vae(vae_model, datagen, 45)
    


    ## VAE-GAN TRAINING 
    # encoder = vae_module.vae_encoder()
    # decoder = vae_module.vae_decoder(encoder.layers[-1].output_shape)
    # vae_model = vae_module.vae(encoder, decoder)

    # discriminator_model = gan_module.discriminator()
    # gan_model = gan_module.gan_model(vae_model, discriminator_model)
    # gan_module.train_gan(gan_model, datagen, epochs=100)


    ## VAE-GAN MODEL TESTING
    encoder = vae_module.vae_encoder()
    decoder = vae_module.vae_decoder(encoder.layers[-1].output_shape)
    vae_model = vae_module.vae(encoder, decoder)
    vae_model.load_weights(f'./model_weight/generator/generator_weight_epoch_1.h5')
    y_pred = vae_model(np.expand_dims(datagen[30], axis=0))
    utils.plot_streamplot_data(y_pred,height=30, save_plot_to=f"./result penting/video/Generated_result_1.mp4", show_figure=False)
    np.save(f"hasil_beta_vae_{i}0.npy", y_pred)