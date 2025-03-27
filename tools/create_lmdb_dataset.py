#!/usr/bin/env python3
"""a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py"""
import io
import os

import fire
import lmdb
import numpy as np
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        img = Image.open(io.BytesIO(imageBin)).convert('RGB')
        return np.prod(img.size) > 0
    except Exception:
        return False


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=250 * 1024 ** 3)

    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as f:
        data = f.readlines()

    nSamples = len(data)
    for i, line in enumerate(data):
        try:
            imagePath, label = line.strip().split(maxsplit=1)
            imagePath = os.path.join(inputPath, imagePath)
            with open(imagePath, 'rb') as f:
                imageBin = f.read()

            if checkValid and not checkImageIsValid(imageBin):
                with open(os.path.join(outputPath, 'error_image_log.txt'), 'a') as log:
                    log.write(f'{i}-th image is invalid or corrupted: {imagePath}\n')
                continue

            imageKey = f'image-{cnt:09d}'.encode()
            labelKey = f'label-{cnt:09d}'.encode()
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print(f'Written {cnt} / {nSamples}')
            cnt += 1

        except FileNotFoundError:
            with open(os.path.join(outputPath, 'error_image_log.txt'), 'a') as log:
                log.write(f'{i}-th image file not found: {imagePath}\n')
        except Exception as e:
            with open(os.path.join(outputPath, 'error_image_log.txt'), 'a') as log:
                log.write(f'{i}-th image data error: {imagePath}, Error: {str(e)}\n')

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print(f'Created dataset with {nSamples} samples')


if __name__ == '__main__':
    fire.Fire(createDataset)
