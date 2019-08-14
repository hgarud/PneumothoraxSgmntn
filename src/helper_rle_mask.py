import csv
import cv2

from mask_functions import rle2mask

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


if __name__ == "__main__":
    get_masks_from_rle('/home/arshita/Desktop/project_2/siim-acr-pneumothorax-segmentation/SIIM/train-rle.csv',
                        '/media/arshita/Windows/Users/arshi/Desktop/Project2/mask_new')
