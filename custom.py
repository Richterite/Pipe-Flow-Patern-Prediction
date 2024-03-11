import tensorflow as tf


def reconstruction_loss(y_true, y_pred):
    """
    Loss function untuk mengetahui tingkat kemiripan 
    data yang dibentuk oleh model dengan data asli

    @Parameter
    *y_true: numpy.array
    *y_pred: numpy.array
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

def kl_div_loss(mu, log_varians):
    """
    Loss function untuk mengetahui seberapa jauh perbedaan distribusi data antara
    distribusi data pada model dengan distribusi data asli

    @Parameter
    *mu : float
    *log_varians : float
    """
    return (-0.5) * tf.reduce_mean(1 + log_varians - tf.square(mu) - tf.exp(log_varians))

def vae_loss(y_true, y_pred, mu, log_var, data_output_shape):
    """
    Loss function untuk menghitung nilai loss dari variational Auto-encoder.
    Loss ini merupakan gabungan dari reconstruction_loss dan KL-div loss.

    @Parameter
    *y_true : numpy.array
    *y_pred : numpy.array
    *mu : float
    *log_varians : float
    *data_output_shape : float
    """
    return reconstruction_loss(y_true, y_pred) + (1/data_output_shape) * kl_div_loss(mu, log_var)