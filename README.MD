[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python->=3.8-blue.svg)](https://www.python.org/downloads/)
# image-similarity
Streamlit app for getting the similar products by photo. List of products and photos are coming from scraping the websites. 
Images are stored in directory and product data stored in MongoDB.
* Note: with big amount of data or long calculating page is refreshing.
### Features
* This project (docker branch) is depending on the other `scrapy-spiders` project, that is scraping products (photos and data).
* Built on back of docker & docker-compose.
* Nginx serves everything. Project will be ready on `port:8080` and it's under basic htpasswd authentication.
* Streamlit port `8501` is not exposed for safety.
* Image similarity is calculated by 4 metrics: PSNR, RMSE, SSIM and SRE. Top 20 results from each metric are chosen.
### First-time setup
1. Set Mongo connection data and path to images in `.env`. Settings are equivalent to the scrapy project. 
Note: IP address of Mongo will be equal to `docker network` address.
2. Add user for Nginx authentication. More details in the README at `nginx/auth`.
3. Build and start project `docker compose up -d`
### TODO
- [ ] Selector for similarity metrics
- [ ] Drop-down list with collections in DB
- [ ] Controlling the comparison process. If page is refreshed or new comparison started, kill the current process. 
Or save the result in file that later can be browsed or downloaded.