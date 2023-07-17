import base64
import cv2 as cv
import heapq
import math
import numpy as np
import os
import pandas as pd
import pillow_avif
import streamlit as st
from io import BytesIO
from multiprocessing import Pool
from image_similarity_measures.quality_metrics import psnr, rmse, ssim, sre
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')

compare_images_path = 'compare_images'
original_images_path = 'original_images'


def process_images(images_path: str) -> None:
    for image in os.listdir(images_path):
        image_path = os.path.join(images_path, image)
        if os.path.isfile(image_path):
            if image_path.endswith('.avif'):
                image_path = convert_avif_to_jpg(image_path)
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

    if height != 400 and width != 400:
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


def convert_avif_to_jpg(image_path: str) -> str:
    avif_img = Image.open(image_path)
    avif_img.save(image_path.replace("avif", 'jpg'), 'JPEG')
    os.remove(image_path)

    return image_path.replace("avif", 'jpg')


def compare_images(comp_img_name) -> dict:
    comp_img_path = os.path.join(compare_images_path, comp_img_name)
    comp_img = cv.imread(comp_img_path)

    return {
        'img_path': comp_img_path,
        'psnr': psnr(original_image, comp_img),
        'ssim': ssim(original_image, comp_img),
        'rmse': rmse(original_image, comp_img),
        'sre': sre(original_image, comp_img)
    }


def get_similar_by_metric(compare_results: dict) -> set:
    result = set()

    for metric, metric_values in compare_results.items():
        if metric == 'rmse':
            result.update(set(heapq.nsmallest(10, metric_values, key=metric_values.get)))
        else:
            result.update(set(heapq.nlargest(10, metric_values, key=metric_values.get)))

    return result


def get_thumbnail(path):
    i = Image.open(path)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


if __name__ == '__main__':
    st.title('Image similarity app')
    st.markdown('---')
    uploaded_image = st.file_uploader('Upload original image', type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_image is not None:
        Image.open(uploaded_image).save('img.png')
        resize_image_to_400px('img.png')
        process_images(compare_images_path)

        total_result = {'psnr': {}, 'ssim': {}, 'rmse': {}, 'sre': {}}
        original_image = cv.imread('img.png')
        st.image(uploaded_image, 'Original image')
        st.markdown('---')
        st.write('## Similar images')
        progress_bar = st.progress(0)
        progress_status = st.empty()
        progress_bar_count = 0
        images_current = 0
        images_count_divider = len(os.listdir(compare_images_path)) / 100

        with Pool(4) as pool:
            for result in pool.imap(compare_images, os.listdir(compare_images_path)):
                images_current += 1
                total_result['psnr'].update({result['img_path']: result['psnr']})
                total_result['ssim'].update({result['img_path']: result['ssim']})
                total_result['rmse'].update({result['img_path']: result['rmse']})
                total_result['sre'].update({result['img_path']: result['sre']})
                if images_current >= images_count_divider * progress_bar_count:
                    progress_bar.progress(progress_bar_count)
                    progress_status.write(str(progress_bar_count) + ' %')
                    progress_bar_count += 1

        cv.waitKey(0)
        cv.destroyAllWindows()
        similar_images = get_similar_by_metric(total_result)
        table_total = pd.DataFrame(total_result)
        table = pd.DataFrame(similar_images, columns=['Path'])
        table['Image'] = table.Path.map(lambda f: get_thumbnail(f))
        st.write(table.to_html(formatters={'Image': image_formatter}, escape=False), unsafe_allow_html=True)
        st.dataframe(table_total)
