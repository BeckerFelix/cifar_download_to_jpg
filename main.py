import os
import cPickle
import numpy as np
from scipy.misc import imsave


# define the ration between train and val data. Images will be stored into corresponding folders
TRAIN_VAL_RATIO = 0.7


def load_batch(filename):
    """Extracts all images and their according class label from given cifar batch

    Arguments:
        filename {[string]} -- [path to the cifar batch to use]

    Returns:
        [tuple] -- [np.array of images and np.array of according labels]
    """

    batch = open(filename, 'rb')
    data_dict = cPickle.load(batch)
    images = data_dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = data_dict['labels']

    return (np.array(images), np.array(labels))


def make_dir_if_not_exists(dirpath):
    """creates a directory on disk if it does not already exist

    Arguments:
        dirpath {[string]} -- [path to the dir that shall be created]
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def main():

    # define the object classes contained in the dataset
    category_names = ['airplane', 'automobile', 'bird', 'cat',
                      'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # create output folders if they do not exist

    for dirname in ('cifar10', 'cifar10/train', 'cifar10/val'):
        make_dir_if_not_exists(dirname)
    for dir_level_1 in ('train', 'val'):
        for dir_level_2 in category_names:
            make_dir_if_not_exists(os.path.join(
                'cifar10', dir_level_1, dir_level_2))

    # download the dataset
    os.system("wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    # extract it from tar.gz file
    os.system("tar -xvzf cifar-10-python.tar.gz")
    # delete the tar.gz file
    os.system("rm cifar-10-python.tar.gz")

    # loop over all the downloaded cifar data batches
    for filename in os.listdir("./cifar-10-batches-py"):
        if filename.startswith("data_batch"):
            images, labels = load_batch(os.path.join(
                "./cifar-10-batches-py", filename))

            # unpack and save data to the folders
            for index, (image, label) in enumerate(zip(images, labels)):
                if index % 10 >= TRAIN_VAL_RATIO*10:
                    train_or_val = 'val/'
                else:
                    train_or_val = 'train/'
                image = image.transpose(1, 2, 0)
                imsave(os.path.join("./cifar10", train_or_val,
                                    category_names[label], str(index) + ".jpg"), image)

    # cleanup the cifar-10-batches-py folder
    os.system("rm -r cifar-10-batches-py")


if __name__ == "__main__":
    main()
