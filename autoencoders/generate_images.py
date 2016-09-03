import argparse

from images_from_text import images_from_file


parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="path to a text file to generate images with")
parser.add_argument("output_dir", help="what it says")
parser.add_argument("--font_path",
                    default='/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf',
                    help="path to .ttf file with the desired font")
parser.add_argument("--font_size", type=int, default=5, help="font size to use")
parser.add_argument("--img_width", type=int, default=200,
                    help='width of the resulting images')
parser.add_argument("--img_height", type=int, default=200,
                    help='height of the resulting images')

args = parser.parse_args()

images_from_file(args.input_path, args.output_dir, args.font_path, args.font_size,
                 args.img_width, args.img_height)