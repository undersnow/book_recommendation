# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
from lxml import etree
import csv
import pandas as pd
from PIL import Image
"""
Created on Mon May 27 15:24:52 2019

@author: think
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:59:07 2019

@author: think
"""

csv_file_name = "test.csv"

with open(csv_file_name, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["title", "author", "url"])


for i in range(4768):
    url = 'http://hrbgy.chineseall.cn/org/show/2542/search/all/' + str(i)
    html = requests.get(url)

    soup = BeautifulSoup(html.text, "lxml")

    div_list = soup.select('div.img')
    title_list = soup.select('div.boxListLi5>h2')
    author_list = soup.select('div.other')

    length = len(div_list)

    for i in range(length):
        link = 'http://hrbgy.chineseall.cn' + div_list[i].a.get('href')
        book_details = requests.get(link)
        book_soup = BeautifulSoup(book_details.text, "lxml")
        author = book_soup.select('div.dgBook_fm_dl>ul>li')
        author_name = author[0].text
        author_name = ' '.join(author_name.split())

        title = title_list[i].a.get('title')
        title = title.replace("/", "_")

        img_link = div_list[i].img.get('src')
        response = requests.get(img_link)
        img = response.content
        img_title = './img/' + title + '.jpg'
        with open(img_title, 'wb') as f:
            f.write(img)
        img = Image.open(img_title)
        new_img = img.resize((128, 256), Image.ANTIALIAS)
        title_img = './new/' + title + '.jpg'
        new_img.save(title_img, quality=100)
        with open(csv_file_name, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([title, author_name, link])
