import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj, transform
from sklearn.cluster import DBSCAN
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from tqdm import tqdm

def df_to_gdf(df, x, y):
    geometry = [Point(xy) for xy in zip(df[x], df[y])] # create Geometry series with lat / longitude
    df = df.drop([x, y], axis=1)
    gdf = gpd.GeoDataFrame(df, crs=None, geometry=geometry)
    return gdf

def get_cluster_centers(clusters):
    coords = list()
    
    for i in clusters[clusters.cluster != -1].cluster.unique():
        center = clusters[clusters.cluster == i].unary_union.centroid.coords[0]
        coord = list(center)
        coords.append(coord)
    return np.array(coords)

def get_DBSCAN_cluster_centers(x, min_samples, eps=1e-3):
    X = x.to_numpy()
    dbscan = DBSCAN(eps, min_samples).fit(X)
    gdf = df_to_gdf(x, x='lng', y='lat')
    gdf['cluster'] = dbscan.labels_
    coords = get_cluster_centers(clusters=gdf)
    return coords

def get_share_urls(element_num=50):
    share_urls = list()
    
    for i in range(element_num):
        driver.switch_to.default_content()
        time.sleep(1.0)
        driver.switch_to.frame('searchIframe')
        time.sleep(1.0)

        try:
            ul_tag = driver.find_element_by_class_name('undefined')
            element = ul_tag.find_elements_by_tag_name('li')[i]
            driver.execute_script('arguments[0].scrollIntoView(true);', element)
            element.find_elements_by_tag_name('a')[0].click()
            time.sleep(1.0)
        except IndexError:
            break

        driver.switch_to.default_content()
        time.sleep(1.0)
        driver.switch_to.frame('entryIframe')
        time.sleep(1.0)

        try:
            place_detail_wrapper = driver.find_element_by_class_name('place_detail_wrapper')
            place_detail_wrapper.find_element_by_id('_btp.share').click()
            time.sleep(1.0)

            spi_list = place_detail_wrapper.find_element_by_xpath('/html/body/div[4]/ul/li/ul')
            copy_url = spi_list.find_elements_by_tag_name('li')[-1]
            share_urls.append(copy_url.find_element_by_tag_name('a').get_attribute('href'))
        except:
            continue
    return share_urls

if __name__ == "__main__":
    DATA_DIR = './data/pre_stop_points.csv'
    stop_df = pd.read_csv(DATA_DIR)
    edu_df = stop_df[stop_df.activity == 3]
    centroids = get_DBSCAN_cluster_centers(edu_df[['lng', 'lat']], 1)

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('--lang=ko')
    options.add_argument('window-size=1920,1080')

    driver = webdriver.Chrome('./chromedriver', options=options)
    driver.implicitly_wait(3.0) # seconds

    sub_category_names = ['학교', '학원', '유치원', '어린이집']
    share_urls = list()

    for location in tqdm(centroids):
        lat = location[1]
        lng = location[0]
        center = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), lng, lat)
        zoom_level = 15
        url = f'https://map.naver.com/v5/?c={center[0]},{center[1]},{zoom_level},0,0,0,dh'
        
        for sub_category in sub_category_names:
            driver.get(url)
            driver.maximize_window()

            input_box = driver.find_element_by_class_name('input_box')
            input_box.find_element_by_tag_name('input').send_keys(sub_category)
            input_box.find_element_by_tag_name('input').send_keys(Keys.RETURN)
            time.sleep(1.0)

            cat_share_urls = get_share_urls()
            share_urls.extend(cat_share_urls)

    driver.close()
    share_url_df = pd.DataFrame(share_urls, columns=['share_urls'])
    share_url_df.to_csv('./data/edu_share_urls.csv', index=False)