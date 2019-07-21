import os
import argparse
import subprocess

def bash(command):
    subprocess.run(command.split())

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default="train")
args = parser.parse_args()

base_dir = "/media/hrishi/OS/1Hrishi/2Kaggle/Pneumothorax_Segmentation/data/SIIM"
output_dir = "/media/hrishi/OS/1Hrishi/2Kaggle/Pneumothorax_Segmentation/data/SIIM/png-images-" + args.type
if ~os.path.exists(output_dir):
    os.mkdir(output_dir)


if args.type == "train":
    images_dir = sorted(os.listdir(base_dir))[1]
    images_dir_path = os.path.join(base_dir, images_dir)
    img_list = os.listdir(images_dir_path)
    for img_dir in img_list:
        image_path1 = os.path.join(images_dir_path, img_dir)
        dcm_image_path = os.path.join(image_path1, os.listdir(image_path1)[0])
        dcm_image = os.listdir(dcm_image_path)
        bash_cmd = "dcmj2pnm -v +fo +cg +on " + str(os.path.join(dcm_image_path, dcm_image[0])) + " " + str(os.path.join(output_dir, dcm_image[0][:-4])) + ".png"
        bash(bash_cmd)
elif args.type == "test":
    images_dir = sorted(os.listdir(base_dir))[0]
    images_dir_path = os.path.join(base_dir, images_dir)
    img_list = os.listdir(images_dir_path)
    for img_dir in img_list:
        image_path1 = os.path.join(images_dir_path, img_dir)
        dcm_image_path = os.path.join(image_path1, os.listdir(image_path1)[0])
        dcm_image = os.listdir(dcm_image_path)
        bash_cmd = "dcmj2pnm -v +fo +cg +on " + str(os.path.join(dcm_image_path, dcm_image[0])) + " " + str(os.path.join(output_dir, dcm_image[0][:-4])) + ".png"
        bash(bash_cmd)
