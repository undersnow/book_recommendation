# -*- coding: utf-8 -*-
"""

姓名:熊俊
文件描述：爬取交大思源探索书本信息和allitebooks网站编程相关书籍信息

"""


import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re

import  requests 
import aiofiles
import time
from PIL import Image
import csv


class fetchbook:
    """fetch book information and img

    Crawling book information from allitebooks and SJTULIB(思源探索)in an asynchronous way

    Attributes:
        flag: A boolean indicating if we crawling allitebooks or SJTULIB
        url: A base url string
        baseurl: A url set where we can crawl all information

    """
    def __init__(self,url,baseurl,flag):
        self.url=url
        self.baseurl=baseurl
        self.flag=flag
        if flag:
            for i in range(1,3):
                self.baseurl.append(self.url+str(i))
        else:
            for i in range(1, 3):
                self.baseurl.append(self.url+str(i))
    
    async def parseallitebooks(self,html):
        """parse html from allitebooks. """
        soup = BeautifulSoup(html, "lxml")
        book_list = soup.select("article")
        imgurlbase = "img/"
        newimgurlbase='./new/'
        for book in book_list:
            title= book.select_one(".entry-title").get_text().replace("/", "_")
            url=book.select_one(".entry-title").a.get('href')
            imgurl= book.select_one('.attachment-post-thumbnail').get('src')
            author= book.select_one(".entry-author").get_text()[3:]
            d = []
            d.append(title)
            d.append(author)
            d.append(url)

            writer.writerow(d)
            title_img = imgurlbase + title + ".jpg"
            new_title_img=newimgurlbase+title+'.jpg'
            async with aiohttp.ClientSession() as session:
                content=await self.fetch(imgurl,session)
            with open(title_img, 'wb') as fp:
                fp.write(content)
            #change img size.
            img = Image.open(title_img)
            img=img.convert('RGB')
            new_img = img.resize((128, 256), Image.ANTIALIAS)
            new_img.save(new_title_img, quality=100)

    async def parsesjtu(self,html):
        """parse html from SJTULIB"""
        soup = BeautifulSoup(html, "html5lib")
        book_list = soup.select("article")
        imgurlbase = "img/"
        newimgurlbase='./new/'
        book_list = soup.find_all(class_="favorite")
        for j in book_list:
            imgurl = j.span.get('logo')
            title = j.span.get('title').replace("/", "_")
            author = j.span.get('author')
            appurl = j.span.get('appurl')
            d = []
            d.append(title)
            d.append(author)
            d.append(appurl)
            writer.writerow(d)
            title_img = imgurlbase + title + ".jpg"
            new_title_img=newimgurlbase+title+'.jpg'
            async with aiohttp.ClientSession() as session:
                content=await self.fetch(imgurl,session)
            with open(title_img, 'wb') as fp:
                fp.write(content)
            #change img size
            img = Image.open(title_img)
            new_img = img.resize((128, 256), Image.ANTIALIAS)
            new_img.save(new_title_img, quality=100)

    async def fetch(self,url,session):
        """get image content."""
        async with session.get(url) as response:
            return await response.read()


    async def crawl(self,url,session):
        """get html infomation."""
        r=await session.get(url)
        html=await r.text()
        return html

    async def runit(self,loop):  
        """The main entry for asynchronous access to information""" 
        async with aiohttp.ClientSession() as session:
            tasks=[loop.create_task(self.crawl(url,session)) for url in self.baseurl]
            finished,unfinished=await asyncio.wait(tasks)
            htmls=[f.result() for f in finished]
            if self.flag:
                for i in htmls:
                    await self.parseallitebooks(i)
            else:
                for i in htmls:
                    await self.parsesjtu(i)

def main():

    sjtubase = 'http://fx.sjtulib.superlib.net/s?strchannel=11%2C12&strtype=2&size=15&isort=0&x=788_1071&pages='
    basesjtu=[]
    sjtu=fetchbook(sjtubase,basesjtu,0)
    csvFile = open('data.csv', 'a', newline='', encoding='utf_8')
    writer = csv.writer(csvFile)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(sjtu.runit(loop))
    allitbooksbase='http://www.allitebooks.com/page/'
    baseallit=[]
    allit=fetchbook(allitbooksbase,baseallit,1)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(allit.runit(loop))

if __name__ == "__main__":
    main()
