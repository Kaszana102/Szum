from math import sin, cos, radians
import os
import random
from PIL import Image, ImageDraw, ImageOps


def augment(input_path: str, flip: bool, angle: float, target_size: tuple[int, int]) -> Image:
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

    #final_image.save(output_path)
    return final_image



def augment_all_images(input_dir: str, output_dir: str, max_angle: int, target_size: tuple[int, int],
                       variants: int) -> None:
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
            image = augment(os.path.join(input_dir, file), flip, angle, target_size)
            image.save(os.path.join(output_dir, f"{i}_{file}"))
        print(f"{index + 1}/{len(files)} {file}")
