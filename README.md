# Semi-compressed pipe flow pattern prediction based on Adversarial Autoencoder
Final Project for physics undergraduate degree based on Generative Adversarial Network Architecture using Tensorflow python Library that can imitate numerical data generated from Navier-Stokes solution solved by Numerical Method and staggered-grid model.


## :wrench: Requirement (Tested on)
Python Version 3.8.10
| Depedencies  | Version | Description |
| ------------- | ------------- |  -------------  |
| Matplotlib  | 3.7.4  | Visualizing the result with graph  |
| Numpy  | 1.24.3  | Used for numerical calculation purpose  |
| Pandas  |2.0.3 | Mutating and manipulating dataset  |
| Pillow  | 10.1.0  | Picture file generating or manipulating picture related  |
| Tensorflow  | 2.13.0 | Back-end computation tools for model training  |
| Keras  | 2.13.1  | Deep learning Framework for creating Model  |
| Cuda  | 11.8  | Improve model training speed by using GPU utility training  |

## :file_folder: File Description

| Filename  | Description |
| ------------- | ------------- | 
| `main.py`  | Main Program file to initiate the program |
| `config.py`  | File for defining constant value for calculating, training, and dataset shape  |
| `custom.py`  | Custom training function file for model training purpose |
| `data_generator.py`  | Generating numerical dataset of Navier-stokes solution and convert to csv or picture  |
| `gan_module.py`  | Generative Adversarial Network (GANs) model Architecture and model training |
| `utils.py`  | Utility functions |


## :pushpin: Dataset and Model Weight
### Dataset [:globe_with_meridians:](https://drive.google.com/drive/folders/1ItIHJcdPWXIA1-xKdYyM3NiVqae_03ME?usp=sharing)
The dataset consist of dynamic solution for Navier-Stokes solution by numerical method and staggered-grid model with 16 timestep. The constant used for Navier-stokes variable value refer to `config.py`.


### Model Weight [:globe_with_meridians:](https://drive.google.com/drive/folders/14KPfD5oTiVvT_c-Q_hJh5jFBFfS_kYHL?usp=sharing)
The weight result of model training with 2 session of training (each 750 epochs). 
* **First run** is the first/initiation of training session and it's weight result used for next session. this training session run for 750 epochs
* **Second run** is the second training session, First run weight result used as it's initial weight. this training session run for 750 epochs






