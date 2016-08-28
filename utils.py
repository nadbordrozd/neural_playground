from keras.models import load_model as keras_load_model
import os
import json

LOG_PATH = "train_log.log"
BULK = 'model.'
CONFIG = "config.json"
ARCHITECTURE = "architecture.json"
WEIGHTS = "weights.h5"

def make_sure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model,  directory, overwrite=False):
    make_sure_dir_exists(directory)
    model.save(os.path.join(directory, BULK))

    config = model.additional_config if hasattr(model, "additional_config") else {}
    with open(os.path.join(directory, CONFIG), "wb") as out:
        out.write(json.dumps(config))

    model.save_weights(os.path.join(directory, WEIGHTS), overwrite=overwrite)


def load_model(directory):
    model = keras_load_model(os.path.join(directory, BULK))
    with open(os.path.join(directory, CONFIG), "rb") as conf:
        config = json.load(conf)
    
    model.additional_config = config
    return model

import logging
# create logger
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG,
                    format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("logginho")
logger.setLevel(logging.DEBUG)

logger.info("======================= START ===========================")