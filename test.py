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
np.set_printoptions(threshold=sys.maxsize)


def vis_parsing_maps(im, parsing_anno, mask, iou, part_of_img, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
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
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.7, vis_parsing_anno_color, 0.3, 0)

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
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='old_people/', annotation_path='./annotations/',  cp='79999_iter.pth'):
    if not os.path.exists(respth):
        os.makedirs(respth)

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
    ious = []
    parts_of_img = []
    excessive_parts = []
    not_found_parts = []
    resolutions = []
    size = 512
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            print('Image {} processed successfully.'.format(image_path))
            mask = Image.open(osp.join(annotation_path, image_path))
            mask = mask.resize((size, size), Image.BILINEAR)
            mask = np.array(mask)


            img = Image.open(osp.join(dspth, image_path))
            resolution = img.size
            image = img.resize((size, size), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            #parsing = out[:, :2].squeeze(0).cpu().numpy().argmax(0)
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            for row_id in range(parsing.shape[0]):
                for col_id in range(parsing.shape[1]):
                    if parsing[row_id][col_id] in (1, 2, 3, 4, 5, 6, 10, 11, 12, 13):
                        parsing[row_id][col_id] = 1
                    else:
                        parsing[row_id][col_id] = 0

            intersection = 0
            union = 0
            part_of_img = 0
            excessive_part = 0
            not_found_part = 0
            for row_id in range(parsing.shape[0]):
                for col_id in range(parsing.shape[1]):
                    if mask[row_id][col_id] == 1:
                        part_of_img += 1
                    if parsing[row_id][col_id] == 1 and mask[row_id][col_id] == 1:
                        intersection += 1
                    if parsing[row_id][col_id] == 1 or mask[row_id][col_id] == 1:
                        union += 1
                    if parsing[row_id][col_id] == 1 and mask[row_id][col_id] == 0:
                        excessive_part += 1
                    if parsing[row_id][col_id] == 0 and mask[row_id][col_id] == 1:
                        not_found_part += 1
            iou = intersection/union
            ious.append(iou)
            parts_of_img.append(part_of_img/(size**2))
            excessive_parts.append(excessive_part/union)
            not_found_parts.append(not_found_part/union)
            resolutions.append(resolution)

            vis_parsing_maps(image, parsing, mask, iou, part_of_img/(size**2), stride=1, save_im=True, save_path=osp.join(respth, image_path))
    return ious, parts_of_img, excessive_parts, not_found_parts, resolutions






if __name__ == "__main__":
    print(torch.cuda.current_device())
    evaluate(dspth='old_people/', annotation_path='./annotations/', cp='79999_iter.pth')


