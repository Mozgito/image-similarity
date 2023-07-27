import base64
import cv2 as cv
import heapq
import math
import numpy as np
import os
import pandas as pd
import pillow_avif
import pymongo
import streamlit as st
from io import BytesIO
from multiprocessing import Pool
from image_similarity_measures.quality_metrics import psnr, rmse, ssim, sre
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')

COMPARE_IMAGES_PATH = ''
MONGO_URI = ''
MONGO_DB = ''


@st.cache_resource
def init_dbconnection():
    return pymongo.MongoClient(MONGO_URI)


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
    comp_img_path = os.path.join(COMPARE_IMAGES_PATH, comp_img_name)
    comp_img = cv.imread(comp_img_path)

    return {
        'img_name': comp_img_name,
        'psnr': psnr(original_image, comp_img),
        # 'ssim': ssim(original_image, comp_img),
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
    i = Image.open(COMPARE_IMAGES_PATH + path)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def image_base64_formatter(im):
    return f'data:image/jpeg;base64,{image_base64(im)}'


def make_clickable(link):
    return f'<a target="_blank" href="{link}">{link}</a>'


def calculate_similarity() -> dict:
    total_result = {'psnr': {}, 'rmse': {}, 'sre': {}}
    with Pool(4) as pool:
        result = pool.map_async(compare_images, os.listdir(COMPARE_IMAGES_PATH))
        for value in result.get():
            total_result['psnr'].update({value['img_name']: value['psnr']})
            #total_result['ssim'].update({value['img_name']: value['ssim']})
            total_result['rmse'].update({value['img_name']: value['rmse']})
            total_result['sre'].update({value['img_name']: value['sre']})
        pool.close()
        pool.join()

    cv.waitKey(0)
    cv.destroyAllWindows()

    return total_result


if __name__ == '__main__':
    st.set_page_config(layout='wide')
    st.title('Image similarity app')
    st.markdown('---')
    client = init_dbconnection()
    database = client[MONGO_DB]
    uploaded_image = st.file_uploader('Upload original image', type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_image is not None:
        Image.open(uploaded_image).save('img.png')
        resize_image_to_400px('img.png')
        process_images(COMPARE_IMAGES_PATH)
        original_image = cv.imread('img.png')
        st.image(uploaded_image, 'Original image')

        st.markdown('---')
        st.write('## Similar images')
        total_similarity_result = calculate_similarity()
        top_similar_images = get_similar_by_metric(total_similarity_result)

        table_data = []
        for image_path in top_similar_images:
            product_data = database['bags'].find({'images.path': 'full/' + image_path})
            for row in product_data:
                table_data.append({
                    'Image': row['images'][0]['path'][5:],
                    'Name': row['name'],
                    'Price': row['price'],
                    'Link': row['url'],
                    'Site': row['site']
                })

        # table = pd.DataFrame(table_data)
        # table['Image'] = table['Image'].map(lambda f: get_thumbnail(f))
        # table['Link'] = table['Link'].apply(make_clickable)
        # st.write(table.to_html(formatters={'Image': image_formatter}, escape=False), unsafe_allow_html=True)

        tableSt = pd.DataFrame(table_data)
        tableSt['Image'] = tableSt['Image'].apply(image_base64_formatter)
        st.data_editor(tableSt, column_config={'Image': st.column_config.ImageColumn(), 'Link': st.column_config.LinkColumn(disabled=True)})
