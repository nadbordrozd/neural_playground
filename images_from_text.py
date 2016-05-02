import os
import PIL
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw

PIXEL_ON = 0  # PIL color to use for "on"
PIXEL_OFF = 200  # PIL color to use for "off"


def make_img(lines, font, spacing, height, width):
    image = PIL.Image.new('L', (width, height), color=PIXEL_OFF)
    draw = PIL.ImageDraw.Draw(image)
    vertical_position = 5
    horizontal_position = 5
    line_spacing = int(round(max_height * 0.8))  # reduced spacing seems better
    lines_in_img = height/line_spacing

    for line in lines:
        draw.text((horizontal_position, vertical_position),
                  line, fill=PIXEL_ON, font=font)
        vertical_position += line_spacing
    c_box = PIL.ImageOps.invert(image).getbbox()
    image = image.crop(c_box)
    return image

def make_images_from_file(input_path, font, height, width):
    test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    line_spacing = int(round(max_height * 0.8))  # reduced spacing seems better
    lines_in_img = height/line_spacing
    
    images = []
    current_lines = []
    with open(input_path, "rb") as lines:
        for line in lines:
            current_lines.append(line)
            if len(current_lines) == lines_in_img:
                yield make_img(current_lines, font, line_spacing, height, width)
                current_lines = []

def make_buncha_imgs(input_paths, output_dir, font, height, width, pic_limit=10):
    i = 0
    for path in input_paths:
        for img in make_images_from_file(path, font, height, width):
            img.save(os.path.join(output_dir, "img_%s.png" % i))
            i += 1
            if i >= pic_limit:
                return
            
