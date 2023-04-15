import os
import sys
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing as mp
import cv2
import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import shutil


def adjust_bounding_box(image, x, y, w, h):
    max_side = max(w, h)
    if w < max_side:
        # try to expand width in equal amounts on both sides
        delta = max_side - w
        left_delta = delta // 2
        right_delta = delta - left_delta
        if x - left_delta < 0:
            # extend more on right side
            right_delta = right_delta + left_delta - x
            left_delta = x
        elif x + w + right_delta > image.shape[1]:
            # extend more on left side
            left_delta = left_delta + right_delta - (image.shape[1] - (x + w))
            right_delta = image.shape[1] - (x + w)
        x = x - left_delta
        w = w + delta
    if h < max_side:
        # expand height
        delta = max_side - h
        top_delta = delta // 2
        bottom_delta = delta - top_delta
        if y - top_delta < 0:
            # extend more on bottom side
            bottom_delta = bottom_delta + top_delta - y
            top_delta = y
        elif y + h + bottom_delta > image.shape[0]:
            # extend more on top side
            top_delta = top_delta + bottom_delta - (image.shape[0] - (y + h))
            bottom_delta = image.shape[0] - (y + h)
        y = y - top_delta
        h = h + delta
    return x, y, w, h



# threadsafe function to create a directory
def create_folder(lock, foldername, nocheck=False):
    with lock:
        if os.path.exists(foldername) and not nocheck:
            shutil.rmtree(foldername)  # delete the entire directory tree

        if not os.path.exists(foldername):
            os.makedirs(foldername)
    return


def diff_images(img1, img2, x, y, w, h):
    # Load the images
    # img1 = Image.open('image1.png')
    # img2 = Image.open('image2.png')

    # Get the difference between the two images
    diff_image = ImageChops.difference(img1[y:y+h, x:x+w], img2[y:y+h, x:x+w])

    # Save the difference image
    # diff_image.save('diff.png')

    # Convert the difference image to a numpy array
    diff_array = np.asarray(diff_image)

    # Get the coordinates of all differing pixels
    coords = np.where(diff_array != 0)
    # print(coords)


def save_image(lock, image_path, image, size, expnum=0):
    # Save the image
    output_dir = os.path.dirname(image_path).replace(f'/{size}/', f'/{size}_cropped_exp_{expnum}/')

    if lock is not None:
        write_dir = output_dir
        create_folder(lock, write_dir, True)
    else:
        return

    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, output_filename)
    # Resize the cropped image to 512x512 using OpenCV's resize() method
    resized_image = cv2.resize(image, (int(size), int(size)))
    cv2.imwrite(output_path, resized_image)
    return
    image.save(output_path)


def crop_mammogram(image_path, threshold=10, mask_size=100, kernel_size=(150, 150)):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert to binary image using a threshold
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Remove small connected components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    num_labels, labeled_image = cv2.connectedComponents(closed_image)
    sizes = np.bincount(labeled_image.ravel())[1:]
    mask = np.zeros_like(binary_image, dtype=bool)
    mask[np.isin(labeled_image, np.flatnonzero(sizes > mask_size))] = True

    # Get the bounding box of the largest connected component (breast region)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Convert the cropped image back to PIL Image
    cropped_image_pil = Image.fromarray(cropped_image)

    return cropped_image_pil


# process_image_1 didnt work
def process_image_1(image_path, size):
    # image = Image.open(image_path)
    # image = image.convert('L')
    # bbox = image.getbbox()
    # cropped_image = image.crop(bbox).resize(image.size)

    cropped_image = crop_mammogram(image_path)
    save_image(image_path, cropped_image, size, 1)


def find_dense_region(image_path, window_size=(50, 50)):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the local average of pixel intensities
    kernel = np.ones(window_size, dtype=np.float32) / (window_size[0] * window_size[1])
    local_avg = cv2.filter2D(image.astype(np.float32), -1, kernel)

    # Find the maximum average value region
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_avg)
    x, y = max_loc
    w, h = window_size

    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Convert the cropped image back to PIL Image
    cropped_image_pil = Image.fromarray(cropped_image)

    return cropped_image_pil


# process_image_2 didnt work
def process_image_2(image_path, size):
    # image = Image.open(image_path)
    # image = image.convert('L')
    # bbox = image.getbbox()
    # cropped_image = image.crop(bbox).resize(image.size)

    cropped_image = find_dense_region(image_path)
    save_image(image_path, cropped_image, size, 2)


def find_effective_region(image_path, size):
    # Load the grayscale mammogram image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Identify the region of interest (ROI) containing the breast tissue
    # For example, using thresholding:
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    breast_contour = max(contours, key=cv2.contourArea)

    # Crop the image to include only the breast tissue
    x, y, w, h = cv2.boundingRect(breast_contour)

    # Adjust the bounding box to a square
    x, y, w, h = adjust_bounding_box(image, x, y, w, h)

    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

    # Resize or rescale the image if necessary
    cropped_image_pil = Image.fromarray(cropped_image)
    # breast_image_resized = cv2.resize(breast_image, (int(size), int(size)))
    return cropped_image_pil


# process_image_3 worked
def process_image_3(lock, image_path, size):
    cropped_image = find_effective_region(image_path, size)
    save_image(lock, image_path, cropped_image, size, 3)


def find_largest_connected_component_bbox(image_path, size):
    threshold=128
    min_area=1000

    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Find the largest connected component, excluding the background (label 0)
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i

    # Check if a connected component was found
    if max_label == 0:
        raise ValueError("No connected components found")

    # Get the bounding box of the largest connected component
    x = stats[max_label, cv2.CC_STAT_LEFT]
    y = stats[max_label, cv2.CC_STAT_TOP]
    w = stats[max_label, cv2.CC_STAT_WIDTH]
    h = stats[max_label, cv2.CC_STAT_HEIGHT]

    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

    cropped_image_pil = Image.fromarray(cropped_image)
    # cropped_image_resized = cv2.resize(cropped_image, (int(size), int(size)))
    return cropped_image_pil
    # return x, y, w, h


# process_image_4 worked
def process_image_4(lock, image_path, size):
    cropped_image = find_largest_connected_component_bbox(image_path, size)
    save_image(lock, image_path, cropped_image, size, 4)


def process_directory(input_dir, size):
    image_paths = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(subdir, file)
                image_paths.append(image_path)

    if procid==3:
        _ = Parallel(n_jobs=mp.cpu_count())(
            delayed(process_image_3)(lock, image_path, size)
            for image_path in image_paths
        )
    elif procid==4:
        _ = Parallel(n_jobs=mp.cpu_count())(
            delayed(process_image_4)(lock, image_path, size)
            for image_path in image_paths
        )
    # for image_path in image_paths:
    #     process_image_4(image_path, size)

if __name__ == '__main__':
    # Debug
    # size='1024'

    # procid = int(sys.argv[1])
    procid = 3 ## Seems like only trick 3 works anyway

    manager = mp.Manager()
    lock = manager.Lock()

    # process_image_3(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/5/1351088028.png', size)
    # process_image_4(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/5/1351088028.png', size)
    # process_image_3(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/10006/1459541791.png', size)
    # process_image_4(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/10006/1459541791.png', size)
    # process_image_3(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/29256/1335458719.png', size)
    # process_image_4(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/29256/1335458719.png', size)
    # process_image_3(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/60047/1833891301.png', size)
    # process_image_4(f'./input/rsna-breast-cancer-detection/train_pngs/output/{size}/60047/1833891301.png', size)

    input_base_dir = './input/rsna-breast-cancer-detection/train_pngs/output'
    sizes = ['256', '512', '768', '1024']

    for size in sizes:
        input_dir = os.path.join(input_base_dir, size)
        # output_dir = os.path.join(output_base_dir, size)
        process_directory(input_dir, size)



# import os
# import glob
# from PIL import Image
# import numpy as np
# from joblib import Parallel, delayed
# import multiprocessing as mp
# from tqdm import tqdm

# def crop_image(image_path, output_dir):
#     # Open the image using PIL
#     image = Image.open(image_path)

#     # Convert image to numpy array
#     image_array = np.array(image)

#     # Find the non-black area (non-zero pixels)
#     non_black_area = np.where(image_array != 0)

#     # Get the bounding box coordinates (min and max values of non-black area)
#     y_min, y_max = np.min(non_black_area[0]), np.max(non_black_area[0])
#     x_min, x_max = np.min(non_black_area[1]), np.max(non_black_area[1])

#     # Crop the image using the bounding box coordinates
#     cropped_image = image_array[y_min:y_max + 1, x_min:x_max + 1]

#     # Save the cropped image
#     output_path = os.path.join(output_dir, os.path.basename(image_path))
#     Image.fromarray(cropped_image).save(output_path)

# def process_images(input_dir, output_dir):
#     # Get all PNG file paths
#     file_paths = glob.glob(f"{input_dir}/*/*.png")

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Process images in parallel
#     # Parallel(n_jobs=mp.cpu_count())(delayed(crop_image)(file_path, output_dir) for file_path in tqdm(file_paths))
#     for file_path in tqdm(file_paths):
#         crop_image(file_path, output_dir)

# if __name__ == "__main__":
#     input_base_dir = "./input/rsna-breast-cancer-detection/train_pngs/output"
#     output_base_dir = "./output/rsna-breast-cancer-detection/train_pngs/output"
#     subdir_names = ["256", "512", "768", "1024"]

#     for subdir in subdir_names:
#         input_dir = os.path.join(input_base_dir, subdir)
#         output_dir = os.path.join(output_base_dir, f"{subdir}_cropped")
#         process_images(input_dir, output_dir)



# import os
# from PIL import Image

# input_dir = './input/rsna-breast-cancer-detection/train_pngs/output/256'
# output_dir = './input/rsna-breast-cancer-detection/train_pngs/output/256_cropped'

# count = 0
# for subdir, dirs, files in os.walk(input_dir):
#     for file in files:
#         # Check if the file is a PNG image
#         if file.endswith('.png'):
#             # Open the PNG image
#             image_path = os.path.join(subdir, file)
#             image = Image.open(image_path)

#             # Convert the image to grayscale
#             image = image.convert('L')

#             # Get the bounding box of the non-black region
#             bbox = image.getbbox()

#             # Crop the image using the bounding box
#             cropped_image = image.crop(bbox)

#             # Save the cropped image
#             output_subdir = subdir.replace('256', '256_cropped')
#             os.makedirs(output_subdir, exist_ok=True)
#             output_path = os.path.join(output_subdir, file)
#             cropped_image.save(output_path)
#             count+=1
#     if count >= 200:
#         break
#
# def find_dense_region(image_path, window_size=(50, 50)):
#     # Load the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Calculate the local average of pixel intensities
#     kernel = np.ones(window_size, dtype=np.float32) / (window_size[0] * window_size[1])
#     local_avg = cv2.filter2D(image.astype(np.float32), -1, kernel)

#     # Find the maximum average value region
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local_avg)
#     x, y = max_loc
#     w, h = window_size

#     # Crop the image using the bounding box
#     cropped_image = image[y:y+h, x:x+w]

#     # Convert the cropped image back to PIL Image
#     cropped_image_pil = Image.fromarray(cropped_image)

#     # Display the cropped image
#     plt.imshow(cropped_image_pil, cmap='gray')
#     plt.show()

#     return cropped_image_pil
