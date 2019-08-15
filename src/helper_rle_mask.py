import argparse
import csv
import cv2
import numpy as np
import os

from mask_functions import rle2mask

# parser = argparse.ArgumentParser()
# parser.add_argument('-t', '--root_dir',
#                     default="/media/arshita/Windows/Users/arshi/Desktop/Project2")
#                     # description="")
# parser.add_argument('-t', '--mask_folder',
#                     default="mask_new3")
#                     # description="")
# parser.add_argument('-t', '--csv_file',
#                     default="train-rle.csv")
#                     # description="")
#
# args = parser.parse_args()



def get_masks_from_rle(csv_file_name, mask_file_name):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    Dict_names={}
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file)
        for row in csv_reader:
            name_img = row[0]
            if name_img not in Dict_names:
                Dict_names[name_img]=1
                present=False
            else:
                Dict_names[name_img]+=1
                present=True

            rle_img = row[1]
            if rle_img!=' -1':
                get_mask = rle2mask(rle_img, 1024, 1024)

                M = cv2.getRotationMatrix2D((512,512), 270, 1.0)
                rotated270 = cv2.warpAffine(get_mask, M, (1024, 1024))

                vertical_img = cv2.flip(rotated270, 1 )
                new2_img_name = mask_file_name + '3/' + name_img  + '.png'

                if present==False:
                    cv2.imwrite(new2_img_name, vertical_img)
                    print("New Image written in folder")

                else:
                    present_img = cv2.imread(new2_img_name)
                    cv2.imwrite(new2_img_name, present_img[:,:,0] + vertical_img)
                    print("Old Image re-written in folder")


def no_pnemo_mask(root):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    imgs_list = list(sorted(os.listdir(os.path.join(root, "SIIM_png_train"))))
    masks_list = list(sorted(os.listdir(os.path.join(root, "mask_new3"))))
    if imgs_list == masks_list:
        print("Equal image and mask files")
    else:
        for img_name in imgs_list:
            if img_name not in masks_list:
                print("Writing blank image")
                create_blank_image = np.zeros((1024,1024,1), np.uint8)
                name = root + "/mask_new3/" + img_name
                cv2.imwrite(name,create_blank_image)


if __name__ == "__main__":
    root = "/media/arshita/Windows/Users/arshi/Desktop/Project2"
    mask_folder = "mask_new3"
    csv_file = "/home/arshita/Desktop/project_2/siim-acr-pneumothorax-segmentation/SIIM/train-rle.csv"
    get_masks_from_rle(csv_file, os.path.join(root, mask_folder))
    no_pnemo_mask(root)

    # get_masks_from_rle(args.csv_file, os.path.join(args.root, args.mask_folder))
    # no_pnemo_mask(args.root)
