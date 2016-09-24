import logging
import os
from glob import glob
import json

LOG_PATH = "train_log.log"
BULK = 'model.'
METADATA = "meta.json"


def make_sure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model,  directory):
    make_sure_dir_exists(directory)
    model.save(os.path.join(directory, BULK))

    meta = model.metadata if hasattr(model, "metadata") else {}
    with open(os.path.join(directory, METADATA), "wb") as out:
        out.write(json.dumps(meta))


def load_model(directory):
    from keras.models import load_model as keras_load_model
    model = keras_load_model(os.path.join(directory, BULK))
    with open(os.path.join(directory, METADATA), "rb") as conf:
        meta = json.load(conf)
    
    model.metadata = meta
    return model


def load_latest_model(directory):
    paths = glob(os.path.join(directory, '*'))
    if paths:
        path = max(paths)
        epoch = int(path[-5:])
        model = load_model(path)
        return model, epoch
    else:
        return None, None


# create logger
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG,
                    format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("logginho")
logger.setLevel(logging.DEBUG)

logger.info("======================= START ===========================")
