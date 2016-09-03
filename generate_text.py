from utils import load_model, load_latest_model
from text_generation import generate_and_print, encode

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="directory where the model was saved")
parser.add_argument("seed", help="seed for generator")
parser.add_argument("chars", type=int,  help="how many characters to generate")
parser.add_argument("diversity", type=float,  help="")
parser.add_argument("--load_latest", type=bool, default=True,
                    help='if true looks for the latest epoch_xxxxx subdirectory '
                         'and loads the model from there. Otherwise looks directly in the'
                         'given directory.')

args = parser.parse_args()

if args.load_latest:
    model, _ = load_latest_model(args.model_path)
else:
    model = load_model(args.model_path)

generate_and_print(model, encode(args.seed), args.diversity, args.chars)