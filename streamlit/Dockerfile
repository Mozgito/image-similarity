FROM python:3.8.10-slim

WORKDIR /streamlit
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

COPY . /streamlit

RUN apt-get update && apt-get install -y libgl1 ffmpeg libsm6 libxext6

RUN pip install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    --no-cache-dir --timeout 60 -r requirements.txt

CMD streamlit run main.py