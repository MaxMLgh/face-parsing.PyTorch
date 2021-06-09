#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import sys
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
np.set_printoptions(threshold=sys.maxsize)


def vis_parsing_maps(im, parsing_anno, mask, iou, part_of_img, stride, save_im=False, save_path='./res/test_res/1.png'):
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)

    im = np.array(im)

    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    index = np.where(vis_parsing_anno == 1)

    vis_parsing_anno_color[index[0], index[1], :] = RED

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(vis_im, 0.7, vis_parsing_anno_color, 0.3, 0)

    # mask
    vis_parsing_mask = mask.copy().astype(np.uint8)
    vis_parsing_mask = cv2.resize(vis_parsing_mask, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_mask_color = np.zeros((vis_parsing_mask.shape[0], vis_parsing_mask.shape[1], 3)) + 255


    index = np.where(vis_parsing_mask == 1)
    vis_parsing_mask_color[index[0], index[1], :] = GREEN

    vis_parsing_mask_color = vis_parsing_mask_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.8, vis_parsing_mask_color, 0.2, 0)

    cv2.putText(vis_im, "IoU:{:.2%} ".format(iou), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.putText(vis_im, "PoI:{:.2%} ".format(part_of_img), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im)

    # return vis_im

def read_net(cp='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return net, to_tensor

def get_parsing(image, to_tensor, net):
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    out = net(img)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
    # classes below are different parts of a person (eye, hair, nose, etc.)
    index_face = np.where((parsing == 1) | (parsing == 2) | (parsing == 3)
                          | (parsing == 4) | (parsing == 5) | (parsing == 6)
                          | (parsing == 10) | (parsing == 11)
                          | (parsing == 12) | (parsing == 13))

    parsing[index_face[0], index_face[1]] = 1
    return parsing

def evaluate(respth='./res/test_res', size=512,  dspth='old_people/', annotation_path='./annotations/',  cp='79999_iter.pth'):
    net, to_tensor = read_net(cp)
    ious = []
    parts_of_img = []
    excessive_parts = []
    not_found_parts = []
    resolutions = []
    with torch.no_grad():
        for image_path in tqdm(os.listdir(dspth)):
            mask = Image.open(osp.join(annotation_path, image_path))
            mask = mask.resize((size, size), Image.BILINEAR)
            mask = np.array(mask)

            img = Image.open(osp.join(dspth, image_path))
            resolution = img.size

            image = img.resize((size, size), Image.BILINEAR)
            parsing = get_parsing(image, to_tensor, net)

            part_of_img = len(np.where(mask == 1)[0])
            intersection = len(np.where(np.logical_and(parsing == 1, mask == 1))[0])
            union = len(np.where(np.logical_or(parsing == 1, mask == 1))[0])
            excessive_part = len(np.where(np.logical_and(parsing == 1, mask != 1))[0])
            not_found_part = len(np.where(np.logical_and(parsing != 1, mask == 1))[0])

            iou = intersection/union
            ious.append(iou)
            parts_of_img.append(part_of_img/(size**2))
            excessive_parts.append(excessive_part/union)
            not_found_parts.append(not_found_part/union)
            resolutions.append(resolution)

            vis_parsing_maps(image, parsing, mask, iou, part_of_img/(size**2), stride=1, save_im=True, save_path=osp.join(respth, image_path))
    return ious, parts_of_img, excessive_parts, not_found_parts, resolutions


def segment(img, size=512,  cp='79999_iter.pth'):
    net, to_tensor = read_net(cp)
    with torch.no_grad():
        resolution = img.size
        image = img.resize((size, size), Image.BILINEAR)
        img_np = np.array(image)
        print('Image processed successfully. Resolution {}.'.format(resolution))
        parsing = get_parsing(image, to_tensor, net)

    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    stride = 1

    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    index = np.where(vis_parsing_anno == 1)
    vis_parsing_anno_color[index[0], index[1], :] = RED

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.7, vis_parsing_anno_color, 0.3, 0)

    return vis_im


if __name__ == "__main__":
    print(torch.cuda.current_device())
    evaluate(dspth='old_people/', annotation_path='./annotations/', cp='79999_iter.pth')


