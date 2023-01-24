import os
import glob
#import path
import cv2
import numpy as np
from collections import OrderedDict
import pickle
from zipfile import ZipFile

def main():
    unzip_mnist_file('./')

    train_path = "./mnist/train/"
    test_path = "./mnist/test/"
    
    train_paths = glob.glob(train_path + '/*/*')
    test_paths = glob.glob(test_path + '/*/*')
    
    train_dataset = read_image_and_label(train_paths)
    test_dataset = read_image_and_label(test_paths)

    save_npy(train_dataset, test_dataset)

    data_dict = read_npy()
    
    save_pickle(data_dict)

    image = data_dict['train_image'][0]

    data_augment(image)


def unzip_mnist_file(paths):
    try:
        if not os.path.exists(paths + 'mnist'):
           with ZipFile(paths + 'mnist.zip') as zipper:
               print('start to unzip mnist.zip file')
               os.makedirs(paths + 'mnist/')
               zipper.extractall(path=(paths + 'mnist/'))
               print('successfully unzip mnist.zip file')
        else:
            print('MNIST directory is alrady exist')
    except OSError:
        print('Error: Fail to unzip')  

def read_image_and_label(paths):
    # TODO: with image folders path, read images and make label with image paths)
    # DO NOT use dataset zoo from pytorch or tensorflow
    images = []
    labels = []
    
    for img_path in paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = img_path.replace('\\','/').split('/')[-2]
        
        images.append(img)
        labels.append(int(label))
       
    return images, labels


def save_npy(train_dataset, test_dataset):
    train_images, train_labels = train_dataset
    test_images, test_labels = test_dataset

    np.save("./train_images.npy",train_images)
    np.save("./test_images.npy", test_images)
    np.save("./train_labels.npy", train_labels)
    np.save("./test_labels.npy", test_labels)

def read_npy():
    # TODO: read npy files and return dictionary
    """
     data = {'train image': [train_images],
             'train label': [train_labels],
             'test_image': [test_images],
             'test_label': [test_labels]
            }
     """
    data_dict = OrderedDict()
    data_dict['train_image'] = np.load("./train_images.npy")
    data_dict['train_label'] = np.load("./train_labels.npy")
    data_dict['test_image'] = np.load("./test_images.npy")
    data_dict['test_label'] = np.load("./test_labels.npy")
        
    return data_dict

def save_pickle(data_dict):
    # TODO: save data_dict as pickle (erase "return 0" when you finish write your code)
    with open('data.pickle', 'wb') as f:
        pickle.dump(data_dict, f)

def data_augment(image):
    # TODO: use cv2.flip, cv2.rotate, cv2.resize and save each augmented image
    cv2.imwrite("./original.jpg",image)
    cv2.imwrite('./horizontal_flip.jpg', cv2.flip(image, 1))
    cv2.imwrite('./vertical_flip.jpg', cv2.flip(image, 0))
    cv2.imwrite('./rotate_90_cw.jpg', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite('./rotate_90_ccw.jpg', cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    cv2.imwrite('./rotate_180.jpg', cv2.rotate(image, cv2.ROTATE_180))
    cv2.imwrite('./resize.jpg', cv2.resize(image, (32, 32)))

if __name__ == "__main__":
    main()
