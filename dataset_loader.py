from math import sin, cos, radians
import os
import random


import cv2
from numpy import asarray
import numpy as np
from PIL import Image, ImageDraw, ImageOps


HORSE = [1, 0, 0, 0]
PENGUIN = [0, 1, 0, 0]
TURTLE = [0, 0, 1, 0]
OTHER = [0, 0, 0, 1]


def load_horses():
    dataset = []

    dir = 'dataset/horses'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, HORSE]
    return dataset


def load_penguins():
    dataset = []

    dir = 'dataset/penguins'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, PENGUIN]
    return dataset


def load_turles():
    dataset = []

    dir = 'dataset/turtles'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, TURTLE]
    return dataset


def load_others():
    dataset = []

    dir = 'dataset/other'
    for filename in os.listdir(dir):
        img = cv2.imread(dir + '/' + filename)
        numpydata = asarray(img)
        dataset += [numpydata, OTHER]
    return dataset


def load_all_dataset():
    '''

    :return: str
    '''

    # return load_horses() + load_penguins() + load_turles() + load_others()
    return load_penguins()


def split1():
    dataset = load_all_dataset()

    # create sets
    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    valid_set = dataset[0::10]
    test_set = dataset[1::10]

    # shuffle
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    x_train_set = [val[0] for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0] for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0] for val in test_set]
    y_test_set = [val[1] for val in test_set]

    # but why?
    x_train_set = np.array(x_train_set)
    x_valid_set = np.array(x_valid_set)
    x_test_set = np.array(x_test_set)

    y_train_set = np.array(y_train_set)
    y_valid_set = np.array(y_valid_set)
    y_test_set = np.array(y_test_set)

    return x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set


def split2():
    dataset = load_all_dataset()

    # create sets
    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    valid_set = dataset[0::10]
    test_set = dataset[1::10]

    # shuffle
    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    x_train_set = [val[0] for val in train_set]
    y_train_set = [val[1] for val in train_set]
    x_valid_set = [val[0] for val in valid_set]
    y_valid_set = [val[1] for val in valid_set]
    x_test_set = [val[0] for val in test_set]
    y_test_set = [val[1] for val in test_set]

    # but why?
    x_train_set = np.array(x_train_set)
    x_valid_set = np.array(x_valid_set)
    x_test_set = np.array(x_test_set)

    y_train_set = np.array(y_train_set)
    y_valid_set = np.array(y_valid_set)
    y_test_set = np.array(y_test_set)

    return x_train_set, x_valid_set, x_test_set, y_train_set, y_valid_set, y_test_set


def augment(input_path: str, output_path: str, flip: bool, angle: float, target_size: tuple[int, int]) -> None:
    """
    Augments the image located at input_path and saves it to output_path.
    The augmentation includes: rotating, flipping, extending borders, and scaling the image.
    :param input_path: path to the image to augment
    :param output_path: path to save the augmented image
    :param flip: should the image be flipped horizontally
    :param angle: angle to rotate the image by
    :param target_size: desired size of the output image. Must be a square
    """

    if target_size[0] != target_size[1]:
        raise AttributeError("Error: target_size must be square.")

    img = Image.open(input_path)

    if flip:
        img = ImageOps.mirror(img)

    new_width = max(img.width, img.height) * 2
    new_height = new_width
    new_size = (new_width, new_height)

    img.thumbnail(new_size, Image.LANCZOS)

    # new_img is a square image, bigger than img, with space for edge extensions
    new_img = Image.new("RGB", new_size)

    left = (new_width - img.width) // 2
    top = (new_height - img.height) // 2
    right = (new_width + img.width) // 2
    bottom = (new_height + img.height) // 2

    new_img.paste(img, (left, top, right, bottom))

    draw = ImageDraw.Draw(new_img)

    # left and right edges
    for y in range(img.height):
        draw.line(((0, y + top), (left, y + top)), img.getpixel((0, y)))
        draw.line(((right, y + top), (new_width, y + top)), img.getpixel((img.width - 1, y)))

    # top and bottom edges
    for x in range(img.width):
        draw.line(((x + left, 0), (x + left, top)), img.getpixel((x, 0)))
        draw.line(((x + left, bottom), (x + left, new_height)), img.getpixel((x, img.height - 1)))

    # corners
    draw.rectangle(((0, 0), (left, top)), fill=img.getpixel((0, 0)))
    draw.rectangle(((right, 0), (new_width, top)), fill=img.getpixel((img.width - 1, 0)))
    draw.rectangle(((0, bottom), (left, new_height)), fill=img.getpixel((0, img.height - 1)))
    draw.rectangle(((right, bottom), (new_width, new_height)), fill=img.getpixel((img.width - 1, img.height - 1)))

    rotated_img = new_img.rotate(angle)

    # calculating dimensions of the largest non-rotated rectangle that can fit inside the rotated image
    angle %= 180
    if angle <= 90:
        inner_width = abs(img.width * cos(radians(angle))) + abs(img.height * sin(radians(angle)))
        inner_height = abs(img.width * sin(radians(angle))) + abs(img.height * cos(radians(angle)))
    else:
        inner_width = abs(img.height * cos(radians(angle))) + abs(img.width * sin(radians(angle)))
        inner_height = abs(img.height * sin(radians(angle))) + abs(img.width * cos(radians(angle)))

    crop_edge = max(inner_width, inner_height)  # so that it's a square

    crop_area = (
        (new_width - crop_edge) // 2,
        (new_height - crop_edge) // 2,
        (new_width + crop_edge) // 2,
        (new_height + crop_edge) // 2
    )

    cropped_img = rotated_img.crop(crop_area)

    final_image = cropped_img.resize(target_size, Image.LANCZOS)

    final_image.save(output_path)


def augment_all_images(input_dir: str, output_dir: str, max_angle: int, target_size: tuple[int, int], variants: int) -> None:
    """
    Calls augment function on all files in the input_dir directory and saves the augmented files to output_dir.
    :param input_dir: directory with the input files
    :param output_dir: directory to save the output
    :param max_angle: maximum angle to rotate the image by (left or right)
    :param target_size: target size of the output images
    :param variants: how many variants of each image to create
    """
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for index, file in enumerate(files):
        for i in range(variants):
            flip = random.choice((True, False))
            angle = random.randint(-max_angle, max_angle)
            augment(os.path.join(input_dir, file), os.path.join(output_dir, f"{i}_{file}"), flip, angle, target_size)
        print(f"{index + 1}/{len(files)} {file}")
