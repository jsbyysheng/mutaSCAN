import cv2
import numpy as np
import uuid
import time
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import os
import json

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt', encoding='utf-8') as handle:
        json.dump(content, handle, indent=4, sort_keys=False, ensure_ascii=False)

def generate_uuid():
    return str(uuid.uuid4())


def generate_datetime():
    return datetime.fromtimestamp(time.time()).strftime(f'%Y%m%d_%H%M%S_%f')


def generate_cache_file(filename, caches_dir='caches', child_dir=None, filetype='', suffix='', prefix=''):
    if child_dir is not None:
        cache = Path(caches_dir) / Path(child_dir)
    else:
        cache = Path(caches_dir)
    cache.mkdir(parents=True, exist_ok=True)
    return cache / Path(
        f"{prefix}{'_' if prefix != '' else ''}{filename}{'_' if suffix != '' else ''}{suffix}{'.' if filetype != '' else ''}{filetype}")


def generate_random_cache_file(caches_dir='caches', child_dir=None, filetype='', suffix='', prefix=''):
    return generate_cache_file(generate_uuid(), caches_dir, child_dir, filetype, suffix, prefix)


def generate_timedate_cache_file(caches_dir='caches', child_dir=None, filetype='', suffix='', prefix=''):
    return generate_cache_file(generate_datetime(), caches_dir, child_dir, filetype, suffix, prefix)

def mutaSCAN_image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized 

def mutaSCAN_image_complementary(image, width = None, height = None):
    complementary_shape = list(image.shape)
    complementary_shape[0] = height
    complementary_shape[1] = width
    complementary = np.zeros(complementary_shape).astype(image.dtype)
    (h, w) = image.shape[:2]
    complementary[0:h, 0:w] = image
    return complementary