import os
import sys
import cv2
from PIL import Image
import numpy as np
import glob
import warnings
import argparse
from cloths_segmentation.pre_trained_models import create_model

def read_and_resize_image(image_path, size, output_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    cv2.imwrite(output_path, img)
    return img

def execute_command(command):
    os.system(command)

def remove_background(img, mask_img):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_img = cv2.erode(mask_img, k)
    img_seg = cv2.bitwise_and(img, img, mask=mask_img)
    back_ground = img - img_seg
    img_seg = np.where(img_seg == 0, 215, img_seg)
    return img_seg, back_ground

def process_images(images, mask_img, background, back_ground=None):
    for img_path in images:
        img = cv2.imread(img_path)
        if background:
            img = cv2.bitwise_and(img, img, mask=mask_img)
            img = img + back_ground
        cv2.imwrite(img_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()

    ori_img = read_and_resize_image("./static/origin_web.jpg", (768, 1024), "./origin.jpg")
    img = read_and_resize_image("origin.jpg", (384, 512), "resized_img.jpg")

    print("Get mask of cloth\n")
    execute_command("python get_cloth_mask.py")

    print("Get openpose coordinate using posenet\n")
    execute_command("python posenet.py")

    print("Generate semantic segmentation using Graphonomy-Master library\n")
    os.chdir("./Graphonomy-master")
    execute_command("python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img")
    os.chdir("../")

    mask_img = cv2.imread('./resized_segmentation_img.png', cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(mask_img, (768, 1024))
    img_seg, back_ground = remove_background(ori_img, mask_img)
    cv2.imwrite("./seg_img.png", img_seg)
    cv2.imwrite('./HR-VITON-main/test/test/image/00001_00.jpg', cv2.resize(img_seg, (768, 1024)))

    execute_command("python get_seg_grayscale.py")

    print("\nGenerate Densepose image using detectron2 library\n")
    execute_command("python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    origin.jpg --output output.pkl -v")
    execute_command("python get_densepose.py")

    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR-VITON-main")
    execute_command("python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test")
   
