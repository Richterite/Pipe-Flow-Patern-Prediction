import numpy as np
import keras
import gan_module
from config import DataShape
import keras

import utils

import keras.backend as K


# Lakukan training ulang dengan memasukkan data kosong 


if __name__ == "__main__":
    ## Cek GPU
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    ## Cek Time and Memory
    # import time
    # import resource

    # Code yang ingin diuji taruh disini

    # time_elapsed = (time.perf_counter() - time_start)
    # memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    # print ("\n\tNumerical Method\nElapsed Time %5.1f secs\nMemory Usage: %5.1f MByte\n\n" % (time_elapsed,memMb))




    # Cek apakah struktur folder sudah tepat
    # Jika folder tidak ada maka buat folder
    # directory = dir(DIR)[:5]
    # for target in directory:
    #     folder = getattr(DIR, target)
    #     if not os.path.isdir(folder):
    #         os.makedirs(folder)
    # Uncomment Jika dibutuhkan
    # datagen = NumericalDataGenerator()
    # datagen.generateData(plot_data=True)



    # Adversarial Autoencoder
    encoder_weight = "./model_weight/generator/generator_encoder_weight_epoch_750_20240424-103622.h5"
    decoder_weight = "./model_weight/generator/generator_decoder_weight_epoch_750_20240424-103622.h5"
    optimizer = keras.optimizers.legacy.RMSprop(momentum=0.2)
    model = gan_module.AdvAutoencoder(optimizer=optimizer)
    del encoder_weight
    del decoder_weight
    del optimizer
    # # model.summary()
    # # model.summary(layer_name="encoder")
    # # model.summary(layer_name="decoder")
    # # model.summary(layer_name="discriminator")




    
    # Load data
    x_data = np.load("./data/data_step16/x_data_16step.npy")
    y_data = np.load("./data/data_step16/y_data_16step.npy")


    height_variance_list = [number for number in range(8, 44)]
    height_datagen = np.array([utils.create_height_matrix(height) for height in height_variance_list])
    datagen = utils.batch_time_series_dataset([x_data, y_data], time_step=DataShape.TIME_STEP, batch_size=DataShape.BATCH_SIZE ,stack_data=True)
    del height_variance_list
    del x_data
    del y_data


    # Train VAE
    model.train_model(dataset=datagen, epochs=750, batch_size=3,save_every=10)
    del model
    del datagen




    # Single Test
    # pressure_variance_list = np.arange(0,5, 0.1)
    # pressure_datagen = np.array([utils.create_pressure_matrix(pressure) for pressure in pressure_variance_list])
    # height_datagen = np.array([utils.create_pressure_matrix(1, height) for height in height_variance_list])

    # # Model recall, precision, f1
    # generator_model = model.build_generator(
    #     encoder_weight='./result penting/generator_encoder_weight_epoch_10_20240425-154117.h5',
    #     decoder_weight='./result penting/generator_decoder_weight_epoch_10_20240425-154117.h5'
    #     )
    # idx = np.random.randint(0, datagen.shape[0], 20)
    # # print(datagen.dtype)
    # y_pred = generator_model.predict(height_datagen[idx]).astype('float64')
    # recall = utils.recall(datagen[idx], y_pred)
    # precision = utils.precision(datagen[idx], y_pred)
    # f1_score = utils.f1_score(datagen[idx], y_pred)
    # print(f"Recall result : {recall}")
    # print(f"Precision result : {precision}")
    # print(f"f1 Score result : {f1_score}")

    # # Model Test
    # for i in range(1, 800 + 1):
    #     CREATE_EVERY = 10
    #     C_TIME = "20240430-122646"
    #     if i % CREATE_EVERY == 0 :
    #         try:
    #             generator_model = model.build_generator(
    #                 encoder_weight=f"./model_weight/generator/generator_encoder_weight_epoch_{i}_{C_TIME}.h5", 
    #                 decoder_weight=f"./model_weight/generator/generator_decoder_weight_epoch_{i}_{C_TIME}.h5"
    #                 )
    #             y_pred = generator_model(np.expand_dims(pressure_datagen[30], axis=0))
    #             np.save(f"hasil_Adv_autoencoder_{i}_epoch_{C_TIME}.npy", y_pred)
    #             utils.plot_streamplot_data(y_pred, height=30, save_plot_to=f"./result penting/video/hasil_Adv_autoencoder_{i}_epoch_{C_TIME}.mp4", show_figure=False)
    #         except Exception as e:
    #             print(e)
    #             print(f"Weight Epoch-{i} Not Found", flush=True)
    #         finally:
    #             K.clear_session()
    # del model
    # del generator_model
    # del y_pred



  
