import cv2 as cv
import math
import numpy as np
import os
from image_similarity_measures.quality_metrics import fsim, psnr, rmse, ssim, sre

np.seterr(divide='ignore', invalid='ignore')

compare_images_path = 'compare_images'
original_images_path = 'original_images'


def resize_images(images_path: str) -> None:
    for image in os.listdir(images_path):
        image_path = os.path.join(images_path, image)
        if os.path.isfile(image_path):
            resize_image_to_400px(image_path)


def resize_image_to_400px(image_path: str) -> None:
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    preferred_height = 400
    preferred_width = 400
    pad_top = 0
    pad_bot = 0
    pad_left = 0
    pad_right = 0

    if height > width:
        preferred_width = round(preferred_height / height * width)
        pad_left = math.floor((400 - preferred_width) / 2)
        pad_right = math.ceil((400 - preferred_width) / 2)

    if height < width:
        preferred_height = round(preferred_width / width * height)
        pad_top = math.floor((400 - preferred_height) / 2)
        pad_bot = math.ceil((400 - preferred_height) / 2)

    if height != 400 & width != 400:
        img_new = cv.resize(img, (preferred_width, preferred_height))
        img_new_padded = cv.copyMakeBorder(
            img_new,
            pad_top,
            pad_bot,
            pad_left,
            pad_right,
            cv.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        cv.imwrite(image_path, img_new_padded)
    cv.waitKey(0)
    cv.destroyAllWindows()


def compare_images(original_image_path: str, compare_images_path: str) -> dict:
    fsim_measures = {}
    psnr_measures = {}
    ssim_measures = {}
    rmse_measures = {}
    sre_measures = {}
    orig_image = cv.imread(original_image_path)

    for image in os.listdir(compare_images_path):
        image_path = os.path.join(compare_images_path, image)
        if os.path.isfile(image_path):
            comp_image = cv.imread(image_path)
            fsim_measures[image_path] = fsim(orig_image, comp_image)
            psnr_measures[image_path] = psnr(orig_image, comp_image)
            ssim_measures[image_path] = ssim(orig_image, comp_image)
            rmse_measures[image_path] = rmse(orig_image, comp_image)
            sre_measures[image_path] = sre(orig_image, comp_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return {'fsim': fsim_measures, 'psnr': psnr_measures,'ssim': ssim_measures, 'rmse': rmse_measures, 'sre': sre_measures}


def get_similar_by_metric(compare_results: dict) -> dict:
    result = {}

    for metric, metric_values in compare_results.items():
        v = list(metric_values.values())
        k = list(metric_values.keys())
        if metric == 'rmse':
            result[metric] = k[v.index(min(v))]
        else:
            result[metric] = k[v.index(max(v))]

    return result


if __name__ == '__main__':
    resize_images(original_images_path)
    resize_images(compare_images_path)
    for image in os.listdir(original_images_path):
        image_path = os.path.join(original_images_path, image)
        if os.path.isfile(image_path):
            compare_result = compare_images(image_path, compare_images_path)
            print(image, compare_result)
            print(image, get_similar_by_metric(compare_result))
