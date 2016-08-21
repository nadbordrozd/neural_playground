from keras.models import model_from_json
import os
import json

LOG_PATH = "train_log.log"
CONFIG = "config.json"
ARCHITECTURE = "architecture.json"
WEIGHTS = "weights.h5"

def make_sure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model,  directory, overwrite=False):
    make_sure_dir_exists(directory)
    json_string = model.to_json()
    with open(os.path.join(directory, ARCHITECTURE), "wb") as out:
        out.write(json_string)
    config = model.additional_config if hasattr(model, "additional_config") else {}
    with open(os.path.join(directory, CONFIG), "wb") as out:
        out.write(json.dumps(model.additional_config))

    model.save_weights(os.path.join(directory, WEIGHTS), overwrite=overwrite)
    
def load_model(directory):
    with open(os.path.join(directory, CONFIG), "rb") as conf:
        config = json.load(conf)
    
    with open(os.path.join(directory, ARCHITECTURE)) as arch_json:
        model = model_from_json(arch_json.read())
    
        
    loss = config['loss']
    optimizer = config['optimizer']
    model.compile(loss=loss, optimizer=optimizer)
    model.additional_config = config
    model.load_weights(os.path.join(directory, WEIGHTS))
    
    return model

import logging
# create logger
logging.basicConfig(filename=LOG_PATH,level=logging.DEBUG, format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("logginho")
logger.setLevel(logging.DEBUG)

logger.info("======================= START ===========================")