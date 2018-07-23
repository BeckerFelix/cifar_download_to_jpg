import os
import cPickle
import random
import numpy as np
from scipy.misc import imsave


# define the ration between train and val data. Images will be stored into corresponding folders
trainVsVal = 0.7

# defines the object classes contained in the dataset
labelNames = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']

# extract all images in one cifar batch
def load_batch(path, filename):

    batch = open(path + filename, 'rb')
    dict = cPickle.load(batch)
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imageArray = np.array(images)
    labelArray = np.array(labels)

    return imageArray, labelArray


def main():
    # create output folders if they do not exist
    if not os.path.exists("./cifar10"):
        os.makedirs("./cifar10")
    if not os.path.exists("./cifar10/train"):
        os.makedirs("./cifar10/train")
    if not os.path.exists("./cifar10/val"):
        os.makedirs("./cifar10/val")
    if not os.path.exists("./cifar10/train/airplane"):
        os.makedirs("./cifar10/train/airplane")
    if not os.path.exists("./cifar10/train/automobile"):
        os.makedirs("./cifar10/train/automobile")
    if not os.path.exists("./cifar10/train/bird"):
        os.makedirs("./cifar10/train/bird")
    if not os.path.exists("./cifar10/train/cat"):
        os.makedirs("./cifar10/train/cat")
    if not os.path.exists("./cifar10/train/deer"):
        os.makedirs("./cifar10/train/deer")
    if not os.path.exists("./cifar10/train/dog"):
        os.makedirs("./cifar10/train/dog")
    if not os.path.exists("./cifar10/train/frog"):
        os.makedirs("./cifar10/train/frog")
    if not os.path.exists("./cifar10/train/horse"):
        os.makedirs("./cifar10/train/horse")
    if not os.path.exists("./cifar10/train/ship"):
        os.makedirs("./cifar10/train/ship")
    if not os.path.exists("./cifar10/train/truck"):
        os.makedirs("./cifar10/train/truck")
    if not os.path.exists("./cifar10/val/airplane"):
        os.makedirs("./cifar10/val/airplane")
    if not os.path.exists("./cifar10/val/automobile"):
        os.makedirs("./cifar10/val/automobile")
    if not os.path.exists("./cifar10/val/bird"):
        os.makedirs("./cifar10/val/bird")
    if not os.path.exists("./cifar10/val/cat"):
        os.makedirs("./cifar10/val/cat")
    if not os.path.exists("./cifar10/val/deer"):
        os.makedirs("./cifar10/val/deer")
    if not os.path.exists("./cifar10/val/dog"):
        os.makedirs("./cifar10/val/dog")
    if not os.path.exists("./cifar10/val/frog"):
        os.makedirs("./cifar10/val/frog")
    if not os.path.exists("./cifar10/val/horse"):
        os.makedirs("./cifar10/val/horse")
    if not os.path.exists("./cifar10/val/ship"):
        os.makedirs("./cifar10/val/ship")
    if not os.path.exists("./cifar10/val/truck"):
        os.makedirs("./cifar10/val/truck")

    # download the dataset
    os.system("wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    # extract it from tar.gz file
    os.system("tar -xvzf cifar-10-python.tar.gz")
    # delete the tar.gz file
    os.system("rm cifar-10-python.tar.gz")

    # loop over all the downloaded cifar data batches
    counter = 0
    for filename in os.listdir("./cifar-10-batches-py"):
        if filename.startswith("data_batch"):
            images, labels = load_batch("./cifar-10-batches-py" + "/", filename)

	    # save data to the folders
            for image, label in zip(images, labels):
                if counter % 10 >= trainVsVal*10:
                    trainOrValDir = 'val/'
                else:
                    trainOrValDir = 'train/'
                filenameImage = str(counter) + ".jpg"
                counter = counter + 1
                image = image.transpose(1, 2, 0)
                imsave("./cifar10/" + trainOrValDir + labelNames[label] +"/"+ filenameImage, image)

    # cleanup the cifar-10-batches-py folder
    os.system("rm -r cifar-10-batches-py")

if __name__ == "__main__":
    main()
