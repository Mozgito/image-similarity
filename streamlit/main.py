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


@st.cache_resource
def init_dbconnection():
    return pymongo.MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)


def resize_image(img_path: str, size: int):
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    preferred_height = size
    preferred_width = size
    pad_top = 0
    pad_bot = 0
    pad_left = 0
    pad_right = 0

    if height > width:
        preferred_width = round(preferred_height / height * width)
        pad_left = math.floor((size - preferred_width) / 2)
        pad_right = math.ceil((size - preferred_width) / 2)

    if height < width:
        preferred_height = round(preferred_width / width * height)
        pad_top = math.floor((size - preferred_height) / 2)
        pad_bot = math.ceil((size - preferred_height) / 2)

    if height != size or width != size:
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
        cv.imwrite(str(size) + img_path, img_new_padded)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return cv.imread(str(size) + img_path)


def compare_images(comp_img_name: str) -> dict:
    if SITE in ['Lazada', 'Shopee']:
        original_image = original_image700
    else:
        original_image = original_image350

    comp_img_path = os.path.join(COMPARE_IMAGES_PATH, COLLECTION, SITE, comp_img_name)
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
            result.update(set(heapq.nsmallest(25, metric_values, key=metric_values.get)))
        else:
            result.update(set(heapq.nlargest(25, metric_values, key=metric_values.get)))

    return result


def image_base64(path: str):
    path = Image.open(COMPARE_IMAGES_PATH + path)
    with BytesIO() as buffer:
        path.save(buffer, 'jpeg')
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def calculate_similarity() -> dict:
    total_result = {'psnr': {}, 'ssim': {}, 'rmse': {}, 'sre': {}}
    with Pool(4) as pool:
        result = pool.map_async(compare_images, os.listdir(COMPARE_IMAGES_PATH + COLLECTION + '/' + SITE))
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
    SITES = database[COLLECTION].distinct('site')

    uploaded_image = st.file_uploader('上传原始图像 / Upload original image', type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_image is not None:
        Image.open(uploaded_image).save('img.jpg')
        original_image700 = resize_image('img.jpg', 700)
        original_image350 = resize_image('img.jpg', 350)
        st.image(uploaded_image, '原始图像 / Original image')

        st.markdown('---')
        st.write('## 类似产品清单 / List of similar products')
        total_similarity_result = {}
        top_similar_images = set()
        table_data = []

        for SITE in SITES:
            total_similarity_result = calculate_similarity()
            top_similar_images.update(get_similar_by_metric(total_similarity_result))

            for image_path in top_similar_images:
                product_data = database[COLLECTION].find({'images.path': COLLECTION + '/' + SITE + '/' + image_path}) \
                    .sort([('price', 1)]).limit(1)
                for row in product_data:
                    if next(filter(lambda d: d.get("链接 / Link") == row['url'], table_data), None) is None:
                        table_data.append({
                            '图像 / Image': COLLECTION + '/' + SITE + '/' + image_path,
                            '产品名 / Name': row['name'],
                            '价格 / Price': row['price'],
                            '货币 / Currency': row['currency'],
                            '链接 / Link': row['url'],
                            '网站 / Site': row['site']
                        })

        tableSt = pd.DataFrame(table_data)
        tableSt['图像 / Image'] = tableSt['图像 / Image'].apply(image_base64)
        st.data_editor(tableSt, use_container_width=True, height=950, column_config={
            '图像 / Image': st.column_config.ImageColumn(),
            '链接 / Link': st.column_config.LinkColumn(disabled=True)
        })
