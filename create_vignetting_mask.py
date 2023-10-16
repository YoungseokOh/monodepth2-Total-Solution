import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
from scipy.optimize import leastsq
from options import MonodepthOptions


class fisheye_dataset_vignetting_tools:
    def __init__(self, options):
        self.file_dir = os.path.dirname(__file__)  # the directory that A5_dataset_splits.py resides in
        self.opt = options
        self.train_val_folder = sorted(listdir(self.opt.data_path))


    # Function to calculate the residuals for least squares circle fit
    def calculate_residuals(self, c, x, y):
        xi = c[0]
        yi = c[1]
        ri = c[2]
        return ((x-xi)**2 + (y-yi)**2 - ri**2)


    def create_vignetting_mask(self):
    # Initialize lists to store the coordinates of the first non-black pixels from left and right for each row
        x_coords = []
        y_coords = []

        non_vignetting_threshold = 33
        inner_circle_margin = 5
        # Random Image selection
        img = cv2.imread("A5_fisheye_image.jpg")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Scan each row of the image
        for i in range(img_gray.shape[0]):

            # Scan from the left
            for j in range(img_gray.shape[1]):
                if np.any(img_gray[i,j] > non_vignetting_threshold):
                    x_coords.append(j)
                    y_coords.append(i)
                    break

            # Scan from the right
            for j in range(img_gray.shape[1]-1, -1, -1):
                if np.any(img_gray[i,j] > non_vignetting_threshold):
                    x_coords.append(j)
                    y_coords.append(i)
                    break

        # Convert the lists to numpy arrays
        x = np.array(x_coords)
        y = np.array(y_coords)

        # Initial guess for circle parameters (center at middle of image, radius half the image width)
        c0 = [img_gray.shape[1]/2, img_gray.shape[0]/2, img_gray.shape[1]/4]

        # Perform least squares circle fit
        c, _ = leastsq(self.calculate_residuals, c0, args=(x, y))

        img_color = img.copy()
        # Draw the circle on the original image
        cv2.circle(img_color, (int(c[0]), int(c[1])), int(c[2])-10, (0, 255, 0), 2)

        # Fill in the inside of the circle
        mask_valid = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        cv2.circle(mask_valid, (int(c[0]), int(c[1])), int(c[2])-inner_circle_margin, 1, -1)
        cv2.imwrite('A5_vignetting_mask.jpg', mask_valid)
        np.save('A5_vignetting_mask.npy', mask_valid)
        print('work done.')


if __name__ == "__main__":
    options = MonodepthOptions()
    tool = fisheye_dataset_vignetting_tools(options.parse())
    tool.create_vignetting_mask()