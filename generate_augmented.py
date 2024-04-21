import os
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def augment(input_path: str, flip: bool, angle: float, target_size: tuple[int, int]) -> Image:
    """
    Augments the image located at input_path and returns it.
    The augmentation includes: rotating, flipping, extending borders, and scaling the image.
    :param input_path: path to the image to augment
    :param flip: should the image be flipped horizontally
    :param angle: angle to rotate the image by
    :param target_size: desired size of the output image. Must be a square
    :return: augmented image
    """

    # Load the image
    img = load_img(input_path, target_size=target_size)
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)

    # Create an ImageDataGenerator object
    datagen = ImageDataGenerator(
        rotation_range=angle,
        horizontal_flip=flip,
        fill_mode='nearest'
    )

    # Perform the image augmentation and return the first generated image
    for batch in datagen.flow(img, batch_size=1):
        return batch[0]


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
