import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths to files, change the path if needed
LABELS_PATH = '../../data/data_imagenet/labels.txt'
DATA_PATH = '../../data/data_imagenet/val/'
IMG_MEAN = [103.939, 116.779, 123.68]  # Mean to subtract from image, used it later, in training script


# Load in images
def load_img(path, new_size = 256):

    """
    Loads in single resized image as array
    
    Args:
        path (string) : path to image file (with extention)
        size (int, int) : size of the shorter dimension

    Returns:
        rgb_im (numpy.ndarray): resized image as an array
    """

    im = Image.open(path)
    
    if im.size[0] < im.size[1]:
        size_perc = new_size/im.size[0]
    else:
        size_perc = new_size/im.size[1]
        
    size = (int(round(im.size[0]*size_perc, 0)), int(round(im.size[1]*size_perc, 0))) # New size of the image

    im = im.resize(size, Image.ANTIALIAS)
    rgb_im = im.convert('RGB')  # Some imageses are in Grayscale
    return np.array(rgb_im, dtype="float32")

# Get center of image array
def center_crop(img_mat, size = (224, 224)):

    """
    Center crops the image array (numpy.array)
    
    Args:
        img_mat (numpy.ndarray) : image as array (h,w,channels)
        size (int, int) : size of cropped image

    Returns:
        img_mat (numpy.ndarray): cropped image
    """

    w,h,c = img_mat.shape
    start_h = h//2-(size[1]//2)  # Size[1] - h of cropped image
    start_w = w//2-(size[0]//2)  # Size[0] - w of cropepd image
    return img_mat[start_w:start_w+size[0],start_h:start_h+size[1], :]


def load_data_imagenet(size = 256, size_crop = (224, 224)):

    """
    Load imagenet validation data and labels
    
    Args:
        size (int, int) : size of initial image after resize
        size_crop (int, int) : size after center cropping the image
    
    Returns:
        (x_val, y_val) : validation data as numpy.ndarray
    """

    # ### Get test and train labels

    # First get all the labels
    y_val = np.loadtxt(fname=LABELS_PATH, dtype="str")  # Loads in tuple ("img_name", class)
    y_val = np.array(list(map(int, y_val[:,1])), dtype="int16")
        
    # ### Load in images as numpy array

    path = DATA_PATH
    val_imgs = [f for f in listdir(path) if isfile(join(path, f))]
    val_imgs = sorted(val_imgs)
    len_val = len(val_imgs)

    # Fill in x_train array with train data

    x_val = np.empty((len_val, size_crop[0], size_crop[1], 3), dtype="float32")

    for i, img_path in enumerate(val_imgs):
        img_mat = load_img(DATA_PATH + img_path, new_size = size)
        x_val[i] = center_crop(img_mat, size = size_crop)  # Crop center of the image

    return (x_val, y_val)
    

def load_data_imagenet_split(size = 256, size_crop = (224, 224), seed = 333):

    """
    Load imagenet validation data and split it into test and validation. In addition subtract mean from each image
    
    Args:
        size (int, int) : size of initial image after resize
        size_crop (int, int) : size after center cropping the image
        seed (int) : random seed
    
    Returns:
        ((x_val, y_val), (x_test, y_test)): validation and test data
    """
    
    x,y = load_data_imagenet(size, size_crop)  # Load in data
    
    x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.5, random_state=seed)
    
    return ((x_val, y_val), (x_test, y_test))    