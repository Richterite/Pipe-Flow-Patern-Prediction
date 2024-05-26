from dataclasses import dataclass


@dataclass
class Const:
    N_POINTS_Y : int = 64
    ASPECT_RATIO : int = 4
    KINEMATIC_VISCOSITY : float = 0.001
    TIME_STEP_LENGTH : float = 0.001
    N_TIME_STEPS : int = 6000
    PLOT_EVERY : int = 375
    STEP_HEIGHT_POINTS : int = 18
    STEP_WIDTH_POINTS : int = 40
    N_PRESSURE_POISSON_ITERATIONS : int = 50
    N_POINTS_X = (N_POINTS_Y ) * ASPECT_RATIO
    CELL_LENGTH = 1.0 / (N_POINTS_Y - 1)




@dataclass
class DataShape:
    TARGET_LENGTH : int = 16
    BATCH_SIZE: int = 36
    TIME_STEP : int = 16
    ROW : int = 64 
    COL : int = 256 
    CHANNEL : int = 2
    OBSTACLE_HEIGHT_VARIETY = [num for num in range(8, 44)]


@dataclass
class DIR:
    LOG: str = "./logs"
    MODEL_WEIGHT_GENERATOR: str = "./model_weight/generator"
    MODEL_WEIGHT_DISCRIMINATOR: str = "./model_weight/discriminator"
    SAVED_GENERATED_FILE: str = "./data/generated_data"
    MODEL_INFO: str = "./model_info"
