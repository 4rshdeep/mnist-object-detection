#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import numpy as np
import shutil
import random
import argparse
from tqdm import tqdm
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--images_num", type=int, default=10000)
parser.add_argument("--image_size", type=int, default=416)
parser.add_argument("--images_path", type=str, default="./images/")
parser.add_argument("--labels_txt", type=str, default="./labels.csv")
parser.add_argument("--small", type=int, default=3)
parser.add_argument("--medium", type=int, default=6)
parser.add_argument("--big", type=int, default=3)
flags = parser.parse_args()

SIZE = flags.image_size

if os.path.exists(flags.images_path): shutil.rmtree(flags.images_path)
os.mkdir(flags.images_path)

image_paths  = [os.path.join(os.path.realpath("."), "./mnist/train/" + image_name) for image_name in os.listdir("./mnist/train")]
image_paths += [os.path.join(os.path.realpath("."), "./mnist/test/"  + image_name) for image_name in os.listdir("./mnist/test")]

def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""

    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return  ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)


def make_image(data, image_path, ratio=1):

    blank = data[0]
    boxes = data[1]
    label = data[2]

    ID = image_path.split("/")[-1][0]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(28*ratio), int(28*ratio)))
    h, w, c = image.shape

    while True:
        xmin = np.random.randint(0, SIZE-w, 1)[0]
        ymin = np.random.randint(0, SIZE-h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.02:
            boxes.append(box)
            label.append(ID)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    # cv2.rectangle(blank, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
    return blank


def process_on_split(idx):
    idxs = split_idxs[idx]
    annotations = []
    for image_num in tqdm(idxs):
        image_path = os.path.realpath(os.path.join(flags.images_path, "%06d.jpg" %(image_num+1)))
        annotation = image_path.split("/")[-1]
        blanks = np.ones(shape=[SIZE, SIZE, 3]) * 255
        bboxes = [[0,0,1,1]]
        labels = [0]
        data = [blanks, bboxes, labels]
        bboxes_num = 0

        # small object
        ratios = [0.5, 0.8]
        N = random.randint(0, flags.small)
        if N !=0: bboxes_num += 1
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], ratio)

        # medium object
        ratios = [1., 1.5, 2.]
        N = random.randint(0, flags.medium)
        if N !=0: bboxes_num += 1
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], ratio)

        # big object
        ratios = [3., 4.]
        N = random.randint(0, flags.big)
        if N !=0: bboxes_num += 1
        for _ in range(N):
            ratio = random.choice(ratios)
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], ratio)

        if bboxes_num == 0: continue
        cv2.imwrite(image_path, data[0])

        boxes = []
        for i in range(len(labels)):
            if i == 0: continue
            xmin = str(bboxes[i][0])
            ymin = str(bboxes[i][1])
            xmax = str(bboxes[i][2])
            ymax = str(bboxes[i][3])
            class_ind = str(labels[i])
            boxes.append([xmin, xmax, ymin, ymax, str(class_ind)])
        for b in boxes:
            writing = annotation + "," + ",".join(b) + "\n"
            annotations.append(writing)
    return annotations


idxs = [i for i in range(flags.images_num)]
split_idxs = np.array_split(idxs, 12)
pool = multiprocessing.Pool(12) 
s = pool.map(process_on_split, range(12))
print("Writing Labels")
with open(flags.labels_txt, "w") as wf:
    wf.write("".join(np.concatenate(s)))
print("Written Labels")