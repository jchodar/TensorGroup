'''
Created on Jan 14, 2017

@author: lukasz
'''


from settings import filepaths as fp
from PIL import Image as im
import numpy as np
import os
from resizeimage import resizeimage as ri

def read_jpg(filepath, weight, height, name):
    image = im.open(filepath)
    image = ri.resize_contain(image, [weight,height])
    image.save('/home/lukasz/Downloads/tmp/'+name)
    image = np.array(image)
    return image

def read_all_jpgs(dir_path, weight, height):
    data = []
    labels = []
    for filename in os.listdir(dir_path):
        labels.append(1 if 'cat' in filename else 0)
        data.append(read_jpg(dir_path+filename, weight, height, filename))
    return np.array(data), np.array(labels)

if __name__ == '__main__':
    read_all_jpgs(fp.dogs_filepath, 200, 200)
    
    