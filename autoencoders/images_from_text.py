import os
import PIL
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw
from glob import glob
import numpy as np
PIXEL_ON = 0  # PIL color to use for "on"
PIXEL_OFF = 255  # PIL color to use for "off"

def img_to_array(img):
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        x = x.reshape((1, x.shape[0], x.shape[1]))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x

def make_img(lines, font, width, height, spacing=0.9):
    image = PIL.Image.new('L', (width, height), color=PIXEL_OFF)
    draw = PIL.ImageDraw.Draw(image)
    vertical_position = 5
    horizontal_position = 5
    line_spacing = int(round(font.size * spacing))  # reduced spacing seems better

    for line in lines:
        draw.text((horizontal_position, vertical_position),
                  line, fill=PIXEL_ON, font=font)
        vertical_position += line_spacing
    return image

def generate_images_from_text(
        text, font_path='/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf',
        img_height=50, img_width=50, font_size=5):
    font = PIL.ImageFont.truetype(font_path, font_size)
    lines = text.split('\n')
    step = 20
    max_lines = 300
    while True:
        for i in range(0, len(lines) - max_lines, step):
            img = make_img(lines[i:i + max_lines], font,  img_width, img_height)
            yield img_to_array(img)


def images_from_file(input_path, output_dir, font_path, font_size, height, width, step=10):
    with open(input_path, "rb") as file:
        lines = file.readlines()
    max_lines = 150
    font = PIL.ImageFont.truetype(font_path, font_size)
    img_num = len(range(0, len(lines) - max_lines, step))
    print 'there will be %s images' % img_num
    for i in range(0, len(lines) - max_lines, step):
        img = make_img(lines[i:i + max_lines], font, width, height)
        filename = "%s_%sx%s.png" % (str(i/step).zfill(5), width, height)
        img.save(os.path.join(output_dir, filename))
        if (i / step) % 100 == 0:
            print 'done %s of %s' % (i / step + 1, img_num)


def images_from_dir(directory):
    paths = glob(os.path.join(directory, '*.png'))

    yield len(paths)
    while True:
        for path in paths:
            yield PIL.Image.open(path)


def train_set_from_dir(directory):
    gen = images_from_dir(directory)
    n = gen.next()
    # yield n
    # yield gen.next()
    images = np.array([img_to_array(gen.next()).flatten() for _ in range(n)])
    return images

