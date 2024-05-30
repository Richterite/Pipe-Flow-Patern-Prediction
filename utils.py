import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data_generator import NumericalDataGenerator
import config


# Setting seed untuk randomisasi yang akan digunakan
np.random.seed(9)

def split_xy_velocity(data):
    """
    Fungsi yang digunakan untuk memisahkan antara komponen aliran 
    fluida yang telah di batch terhadap sumbu-x dan sumbu-y

    @Parameter
    *data : numpy.array | List

    """
    x_vel_data = []
    y_vel_data = []
    for single_data in data:
        x_vel_data.append(single_data[0])
        y_vel_data.append(single_data[1])
    x_vel_data = np.array(x_vel_data)
    y_vel_data = np.array(y_vel_data)

    return x_vel_data, y_vel_data


def conv_out_calc(input_shape, zero_padding, filter_size, stride):
    """
    Digunakan untuk menghitung keluaran yang dihasilkan 
    setelah data dimasukkan kedalam convolution layer

    @Parameter
    *input_shape : List[int,int]
    *zero_padding : int
    *filter_size : int
    *stride: int
    """
    width , height = input_shape
    width_result = (width - filter_size + 2*zero_padding )/stride + 1
    height_result = (height - filter_size + 2*zero_padding )/stride + 1
    if width_result % 1 != 0 or height_result % 1 != 0:
        print("ERROR")
        print(f"Result : Width = {width_result} && Height = {height_result}")
    else:
        print("meet")
        print(f"Result : Width = {width_result} && Height = {height_result}")


def split_and_squeeze_data(data):
    """
    Berfungsi untuk memisahkan komponen kecapatan aliran fluida 
    pada sumbu-x dan sumbu-y, dan kemudian menghilangkan dimensi paling akhir 
    sehingga dapat diproses untuk hal lainnya

    @Parameter
    *data : numpy.array
    """
    x_data, y_data = np.split(data, 2, axis=-1)
    x_data = np.squeeze(x_data, axis=-1)
    y_data = np.squeeze(y_data, axis=-1)

    return x_data, y_data


def batch_time_series_dataset(dataset, time_step, batch_size, stack_data=False):
    """
    Digunakan untuk mengelompokkan data berdasarkan time step data, 
    mengelompokkan data kedalam batch data untuk memudahkan 
    dan meningkatkan kinerja model saat training. 
    Stack Data (opsional) merupakan ya atau tidaknya data kecepatan aliran fluida 
    pada sumbu-x dan sumbu-y ditumpuk kedalam 1 struktur data (numpy array)

    @Parameter
    *dataset : List[numpy.array|List, numpy.array|List]
    *time_step : int
    *batch_szie : int
    stack_data : boolean
    """
    x_data, y_data = dataset
    n_data, height, width = x_data.shape
    assert n_data == y_data.shape[0]

    x_batch = np.zeros((batch_size, time_step, height, width))
    y_batch = np.zeros((batch_size, time_step, height, width))
    for batch_index in range(batch_size):
        timestep_batch = batch_index * time_step
        for time_index in range(time_step):
            x_batch[batch_index, time_index] = x_data[timestep_batch]
            y_batch[batch_index, time_index] = y_data[timestep_batch]
            timestep_batch += 1
    if stack_data:
        stacked_data = np.stack((x_batch, y_batch), axis=-1)
        return stacked_data
    else:
        return x_batch ,y_batch

def train_validation_split(dataset: np.array, train_size,base_model=None, stacked=True):
    """
    Digunakan untuk membuat training set dan validatio set dari dataset. 
    Perbandingan antara jumlah training set dan validation set ditentukan 
    dari rasio yang ada oada parameter train_size.
    base model adalah bentuk dasar dari lingkungan pergerakan fluida. Daerah 
    yang bernilai 1 menandakan bahwa daerah tersebut bukanlah rintangan.
    Sementara daerah yang bernilai 0 menandakan daerah tersebut adalah
    rintangan.

    @Parameter
    *dataset : numpy.array
    *train_size: Float
    base_model: numpy.array
    stacked: boolean
    """
    n, time_step, row, col, channel = dataset.shape
    train_split = int(n * train_size)
    train_dataset = np.zeros((train_split, time_step, row, col, channel))
    validation_dataset = np.zeros((n - train_split, time_step, row, col, channel))
    if base_model != None:
        base_model_train_dataset = np.zeros_like(train_dataset)
        base_model_val_dataset = np.zeros_like(validation_dataset)
    n = np.random.permutation(n)
    val_idx = 0
    if stacked:
        for idx, batch_index in enumerate(n):
            if idx < train_split:
                train_dataset[idx] = dataset[batch_index]
                if base_model != None:
                    base_model_train_dataset[idx] = base_model[batch_index]
            else:
                validation_dataset[val_idx] = dataset[batch_index]
                if base_model != None:
                    base_model_val_dataset[val_idx] = dataset[batch_index] 
                    val_idx += 1 
        del val_idx
        if base_model != None:
            return (train_dataset, base_model_train_dataset), (validation_dataset, base_model_val_dataset)
        else:
            return train_dataset, validation_dataset
        
    else:

        # NOTE : Base model baru disesuaikan sama yang stacked belum yang non-stack
        x_train = dataset[:train_split]
        y_train = dataset[train_split:]

        x_valid = dataset[:train_split]
        y_valid = dataset[train_split:]

        return (x_train, y_train), (x_valid, y_valid)


def create_generator(data, base_model=np.array([])):
    """
    Digunakan untuk membuat sebuah generator yang akan 
    dipakai dalam proses training. Penggunaan generator 
    bertujuan untuk mengurangi beban memori komputer ketika 
    training dilakukan.
    Randomisasi data juga dilakukan agar model dapat mencapai
    generalization.

    @Parameter
    *data: numpy.array
    base_model : numpy.array

    """
    while True:
        if data[0].shape[-1] > 3:
            x, y = data
            n = np.arange(len(x))
            n = np.random.permutation(n)
            for index in n:
                x_gen = x[index]
                y_gen = y[index]
                yield np.expand_dims(x_gen, axis=0), np.expand_dims(y_gen, axis=0)
        else:
            n = np.arange(len(data))
            n = np.random.permutation(n)
            for index in n:
                arr_gen_data = np.expand_dims(data[index], axis=0)
                
                if base_model.any():
                    arr_gen_base = np.expand_dims(base_model[index], axis=0)
                else:
                    arr_gen_base = None
                yield arr_gen_data, arr_gen_base


def create_pressure_matrix(pressure_value):
    """
    Membuat matriks variasi tekanan yang didasarkan kepada lingkungan kerja pipa
    @parameter
    *pressure_value : Int
    """
    # Custom encoder channels = 1, custom decoder channels = 2
    pressure_matrix = np.ones((config.DataShape.TIME_STEP ,config.Const.N_POINTS_Y, config.Const.N_POINTS_X, 2)) * pressure_value
    pressure_matrix[:,:(config.Const.STEP_HEIGHT_POINTS), config.Const.STEP_WIDTH_POINTS] = 0.0
    pressure_matrix[:,config.Const.STEP_HEIGHT_POINTS, :config.Const.STEP_WIDTH_POINTS ] = 0.0
    pressure_matrix[:,:config.Const.STEP_HEIGHT_POINTS, :config.Const.STEP_WIDTH_POINTS ] = 0.0
    # 1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS
    return pressure_matrix


def create_height_matrix(height):
    """
    Membuat matriks variasi ketinggian dengan tekanan konstan disetiap titik
    pada lingkungan kerja pipa

    @parameter
    *heighat : Int
    """
    # Custom encoder channels = 1, custom decoder channels = 2
    pressure_matrix = np.ones((config.DataShape.TIME_STEP ,config.Const.N_POINTS_Y, config.Const.N_POINTS_X, 2))
    pressure_matrix[:,:height, config.Const.STEP_WIDTH_POINTS] = 0.0
    pressure_matrix[:,height, :config.Const.STEP_WIDTH_POINTS ] = 0.0

    # Inside
    pressure_matrix[:,:height, :config.Const.STEP_WIDTH_POINTS ] = 0.0
    # 1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS
    return pressure_matrix

def animation_generator(x_data, y_data):
    """
    Digunakan untuk membuat sebuah 
    generator yang dikhususkan dalam membuat sebuah 
    animasi hasil

    @Parameter
    *x_data : np.array
    *y_data : np.array
    """
    for index in range(len(x_data)):
        x_gen = x_data[index]
        y_gen = y_data[index]
        yield x_gen, y_gen


def match_environment(x_generated,y_generated, height, width=config.Const.STEP_WIDTH_POINTS):
    """
    Digunakan untuk membuat lingkungan kerja pipa dengan
    hambatan tanpa aliran air. Variasi ketinggian rintangan
    diterapkan juga pada fungsi ini.
    
    @parameter
    *x_generated : List
    *y_generated : List
    *height : Int
    width: Int
    """
    x_generated = np.array(x_generated)
    y_generated = np.array(y_generated)
    for index in range(len(x_generated)):
        x_generated[index][:(height + 1),:(width + 1)] = 0.0
        y_generated[index][:(height + 1),:(width + 1)] = 0.0
    
    return x_generated, y_generated


def plot_streamplot_data(dataset, height, title=None, save_plot_to=None, show_figure=True):
    """
    Digunakan untuk memproses data numerik yang kemudian diubah 
    menjadi bentuk visual bergerak (gif/video).
    
    @Parameter
    *dataset: numpy.array
    save_plot_to: String
    show_figure: boolean
    """
    # Pastikan dataset bukanlah data batch
    assert dataset.shape[0] == 1
    dataset = np.squeeze(dataset, axis=0)
    assert dataset.shape[0] == config.DataShape.TIME_STEP

    fig, ax = plt.subplots(figsize=(15, 10))
    obj = NumericalDataGenerator(get_att_only=True)
    x_coor, y_coor = obj.coordinates_x, obj.coordinates_y
    if dataset[0].shape[-1] == 2:
        x, y = split_and_squeeze_data(dataset)
    elif dataset[0].shape[-1] == 1:
        x, y = dataset
        x = np.squeeze(x, axis=-1)
        y = np.squeeze(y, axis=-1)
    else:
        x, y = dataset

    if x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
        y = np.squeeze(y, axis=0)

    x, y = match_environment(x, y, obj.height_var[height])
    generator = animation_generator(x, y)
    print(x.shape, y.shape)
    del obj
    def animate(frame):
        x_gen , y_gen = next(generator)
        ax.clear()
        if title != None:
            ax.set_title(title)
        ax.streamplot(
            x_coor,
            y_coor,
            x_gen,
            y_gen
        )
    
    anim = animation.FuncAnimation(fig, animate, frames=config.DataShape.TIME_STEP-1, repeat=False)
    if save_plot_to != None:
        # Save as video
        video_writer = animation.FFMpegWriter(fps=1)
        anim.save(save_plot_to, writer=video_writer)

        # Save as GIF
        # anim.save(save_plot_to, writer='imagemagick', fps=1)
    if show_figure:
        plt.show()
    plt.close()
    del anim


def create_ones(batch_size,time_step, obstacle=False, obstacle_array=None):
    """
    Digunakan untuk membuat base model dari sebuah data numerik. 
    Apabila daerah base model bernilai 1 maka daerah tersebut bukan merupakan rintangan,
    sementara jika bernilai 0 maka daerah tersebut adalah rintangan.

    @Parameter
    batch_size: int
    time_step : int
    obstacle: boolean
    obstacle_array: List
    """
    base_env_origin = np.ones((batch_size, time_step, config.DataShape.ROW, config.DataShape.COL, config.DataShape.CHANNEL))

    if obstacle:
        assert obstacle_array != None
        for batch_index in range(batch_size):
            # Origin Shape
            base_env_origin[batch_index,:,1:obstacle_array[batch_index] +1, config.Const.STEP_WIDTH_POINTS ,:] = 0.0
            base_env_origin[batch_index,:,:obstacle_array[batch_index], :config.Const.STEP_WIDTH_POINTS,:] = 0.0

            # Custom Shape
            # base_env_custom[batch_index,:,:obstacle_array[batch_index] + 1,:,:] = 0.0
            # base_env_custom[batch_index,:,1:obstacle_array[batch_index] +1, config.Const.STEP_WIDTH_POINTS ,:] = 0.0
            # base_env_custom[batch_index,:,:obstacle_array[batch_index], :config.Const.STEP_WIDTH_POINTS,:] = 0.0


    return base_env_origin

def recall(y_true, y_pred):
    """
    recall adalah matriks evaluasi yang digunakan untuk mengukur
    seberapa sering model mengklasifikasikan true positive dari sampel yang diberikan.
    semakin besar nilai recall maka semakin baik model dalam
    mengidentifikasi true positive dari sampel yang diberikan

    @Parameter
    *y_true : (np.array|List)
    *y_pred : (np.arrat|List)
    """
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0,1)))
    possible_possitive = K.sum(K.round(K.clip(y_true, 0,1)))
    recall_result = true_positive / (possible_possitive + K.epsilon())
    return recall_result

def precision(y_true, y_pred):
    """
    Precision adalah matriks evaluasi yang digunakan untuk mengukur seberapa sering 
    model dapat memprediksi hasil yang benar berdasarkan sampel yang diberikan.
    semakin besar nilai precision maka semakin baik model dalam memprediksi 

    @Parameter
    *y_true : (np.array|List)
    *y_pred : (np.arrat|List)
    """
    true_positive =  K.sum(K.round(K.clip(y_true * y_pred, 0 , 1)))
    predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_result = true_positive / (predicted_positive + K.epsilon())
    return precision_result

def f1_score(y_true, y_pred):
    """
    F1 score adalah sebuah matriks evaluasi yang digunakan sebagai tolak ukur
    baik atau buruknya model, karena F1 score adalah harmonic mean dari recall dan precsion
    Semakin besar nilai F1 score maka kinerja model semakin baik.

    @Parameter
    *y_true : (np.array|List)
    *y_pred : (np.arrat|List)
    """
    precision_result = precision(y_true, y_pred)
    recall_result = recall(y_true, y_pred)
    return 2*((precision_result *recall_result)/ (precision_result +recall_result+K.epsilon()))


def get_flops(model, model_inputs) -> float:
        """
        Menghitung Banyaknya Floation Point Operation pada keras model. Perhitungan menggunakan
        legacy version dari tensorflow v1

        @Parameter
        *model : Keras.Model
        *model_inputs : tf.Constant

        """
        # if not hasattr(model, "model"):
        #     raise wandb.Error("self.model must be set before using this method.")

        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in model_inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        return (flops.total_float_ops)
    
    