from dataclasses import dataclass

@dataclass
class DIR:
    LOG: str = "./logs"
    MODEL_WEIGHT_GENERATOR: str = "./model_weight/generator"
    MODEL_WEIGHT_DISCRIMINATOR: str = "./model_weight/discriminator"
    SAVED_GENERATED_FILE: str = "./data/generated_data"
    MODEL_INFO: str = "./model_info"
