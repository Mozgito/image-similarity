import base64
import cv2 as cv
import heapq
import math
import numpy as np
import os
import pandas as pd
import pymongo
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from multiprocessing import Pool
from image_similarity_measures.quality_metrics import psnr, rmse, ssim, sre
from PIL import Image

load_dotenv()
np.seterr(divide='ignore', invalid='ignore')

COMPARE_IMAGES_PATH = 'compare_images/'
MONGO_URL = os.environ.get("MONGO_URL")
MONGO_DB = os.environ.get("MONGO_DB")
COLLECTION = ''


@st.cache_resource
def init_dbconnection():
    return pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)


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


def compare_images(comp_img_name: str) -> dict:
    comp_img_path = os.path.join(COMPARE_IMAGES_PATH, COLLECTION, comp_img_name)
    comp_img = cv.imread(comp_img_path)

    return {
        'img_name': comp_img_name,
        'psnr': psnr(original_image, comp_img),
        'ssim': ssim(original_image, comp_img),
        'rmse': rmse(original_image, comp_img),
        'sre': sre(original_image, comp_img)
    }


def get_similar_by_metric(compare_results: dict) -> set:
    result = set()

    for metric, metric_values in compare_results.items():
        if metric == 'rmse':
            result.update(set(heapq.nsmallest(20, metric_values, key=metric_values.get)))
        else:
            result.update(set(heapq.nlargest(20, metric_values, key=metric_values.get)))

    return result


def image_base64(path: str):
    path = Image.open(COMPARE_IMAGES_PATH + path)
    with BytesIO() as buffer:
        path.save(buffer, 'jpeg')
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def calculate_similarity() -> dict:
    total_result = {'psnr': {}, 'ssim': {}, 'rmse': {}, 'sre': {}}
    with Pool(4) as pool:
        result = pool.map_async(compare_images, os.listdir(COMPARE_IMAGES_PATH + COLLECTION))
        for value in result.get():
            total_result['psnr'].update({value['img_name']: value['psnr']})
            total_result['ssim'].update({value['img_name']: value['ssim']})
            total_result['rmse'].update({value['img_name']: value['rmse']})
            total_result['sre'].update({value['img_name']: value['sre']})
        pool.close()
        pool.join()

    cv.waitKey(0)
    cv.destroyAllWindows()

    return total_result


if __name__ == '__main__':
    st.set_page_config(layout='wide')
    st.markdown('''
      <style>
        footer {visibility: hidden}
      </style>
    ''', unsafe_allow_html=True)
    st.title('Similar products by photo')
    st.markdown('---')
    client = init_dbconnection()
    database = client[MONGO_DB]
    COLLECTION = 'bags'
    uploaded_image = st.file_uploader('上传原始图像 / Upload original image', type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_image is not None:
        Image.open(uploaded_image).save('img.png')
        resize_image_to_400px('img.png')
        original_image = cv.imread('img.png')
        st.image(uploaded_image, '原始图像 / Original image')

        st.markdown('---')
        st.write('## 类似产品清单 / List of similar products')
        total_similarity_result = calculate_similarity()
        top_similar_images = get_similar_by_metric(total_similarity_result)

        table_data = []
        for image_path in top_similar_images:
            product_data = database[COLLECTION].find({'images.path': COLLECTION + '/' + image_path})
            for row in product_data:
                table_data.append({
                    '图像 / Image': row['images'][0]['path'],
                    '产品名 / Name': row['name'],
                    '价格 / Price': row['price'],
                    '链接 / Link': row['url'],
                    '网站 / Site': row['site']
                })

        tableSt = pd.DataFrame(table_data)
        tableSt['图像 / Image'] = tableSt['图像 / Image'].apply(image_base64)
        st.data_editor(tableSt, use_container_width=True, height=950, column_config={
            '图像 / Image': st.column_config.ImageColumn(),
            '链接 / Link': st.column_config.LinkColumn(disabled=True)
        })