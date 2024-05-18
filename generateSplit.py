import os
import shutil
import random
import generate_augmented
import tensorflow as tf
from distutils.dir_util import copy_tree

ANIMALS = ['horses', 'penguins', 'turtles', 'other']
SETS = ['train', 'valid', 'test']
IMAGES_PER_CLASS = [800, 100, 100]
IMAGE_RESOLUTION = (256, 256)

random.seed(1)


def initialize_directory():
    # check if directory exists
    if not os.path.exists("Splits"):
        os.mkdir("Splits")

    subfolders = ['Split1', 'Split2', 'Split3']
    sets = ['train', 'valid', 'test']
    for subfolder in subfolders:
        split = os.path.join("Splits", subfolder)
        if not os.path.exists(split):
            os.mkdir(split)
            for set_name in sets:
                set_dir = os.path.join(split, set_name)
                os.mkdir(set_dir)
                for animal in ANIMALS:
                    animal_dir = os.path.join(set_dir, animal)
                    os.mkdir(animal_dir)


def split_dataset_into_sets(dataset):
    original_amount = len(dataset)

    random.shuffle(dataset)

    valid_set_size = int(original_amount / 10)

    # create sets
    valid_set = dataset[0:valid_set_size]
    test_set = dataset[valid_set_size:2 * valid_set_size]
    train_set = dataset[2 * valid_set_size:]
    '''
    train_set = [dataset[i] for i in range(len(dataset)) if i % 10 != 0 and i % 10 != 1]
    valid_set = dataset[0::10]
    test_set = dataset[1::10]
    '''
    return train_set, valid_set, test_set


def load_split1():
    """
    loads them as dictionary of sets containing filenames
    split1 like
    :return:
    """
    sets = {
        'train': {},
        'valid': {},
        'test': {}
    }

    for animal in ANIMALS:
        train_set, valid_set, test_set = split_dataset_into_sets(os.listdir(os.path.join("dataset_src", animal)))
        sets['train'][animal] = train_set
        sets['valid'][animal] = valid_set
        sets['test'][animal] = test_set
    return sets


def create_split1(split):
    for _set in SETS:
        dst = 'Splits/Split1'
        for animal, animals in split[_set].items():
            print("split1 " + _set + " " + animal)
            dst_set = os.path.join(dst, _set)
            dst_set = os.path.join(dst_set, animal)
            src_dir = os.path.join("dataset_src", animal)
            for image in animals:
                src_file = os.path.join(src_dir, image)
                dst_file = os.path.join(dst_set, image)
                img = generate_augmented.augment(src_file, False, 0, IMAGE_RESOLUTION)
                tf.keras.utils.save_img(dst_file, img, scale=False)
                # shutil.copy(file, dst_set)


def create_split2(split):
    for _set, image_per_class in zip(SETS, IMAGES_PER_CLASS):
        dst = 'Splits/Split2'
        for animal, images in split[_set].items():
            print("split2 " + _set + " " + animal)
            dst_set = os.path.join(dst, _set)
            dst_set = os.path.join(dst_set, animal)
            src_dir = os.path.join("dataset_src", animal)
            augment = False
            counter = 0

            while True:
                for image in images:
                    counter += 1
                    src_file = os.path.join(src_dir, image)
                    name = os.path.splitext(image)[0]
                    extension = os.path.splitext(image)[1]
                    dst_file = os.path.join(dst_set, name + str(counter) + extension)
                    if augment:
                        if _set == SETS[0]:  # 'training'
                            img = generate_augmented.augment(src_file, True, 60, IMAGE_RESOLUTION)
                            tf.keras.utils.save_img(dst_file, img, scale=False)
                        else:
                            counter = image_per_class  # finish loop
                    else:
                        img = generate_augmented.augment(src_file, False, 0, IMAGE_RESOLUTION)
                        tf.keras.utils.save_img(dst_file, img, scale=False)
                    if counter == image_per_class:
                        break
                if counter == image_per_class:
                    break
                augment = True


def create_split3():
    # copy directory
    copy_tree('Splits/Split2', 'Splits/Split3')

    # use files from split2 and copy to split3
    train_dir = 'Splits/Split3/train'
    valid_dir = 'Splits/Split3/valid'
    for animal in ANIMALS:
        print("split3: " + animal)
        animal_src_dir_ = os.path.join(train_dir, animal)
        animal_dst_dir_ = os.path.join(valid_dir, animal)
        i = 0
        counter = 0
        for file in os.listdir(animal_src_dir_):
            if i % 10 == 0:
                src_file = os.path.join(animal_src_dir_, file)

                name = os.path.splitext(file)[0]
                extension = os.path.splitext(file)[1]
                dst_file = os.path.join(animal_dst_dir_, name+"_" + str(counter) + extension)
                shutil.copyfile(src_file, dst_file)
            i += 1
            counter += 1


initialize_directory()
split1 = load_split1()
print("started split1")
create_split1(split1)
print("started split2")
create_split2(split1)
print("started split3")
create_split3()
