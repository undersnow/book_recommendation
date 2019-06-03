from bs4 import BeautifulSoup
import requests
from lxml import etree
import csv
import pandas as pd
from PIL import Image


def main():
    csvFile = open('data.csv', 'a', newline='', encoding='utf_8')
    fileHeader = ["title", "author", "url"]
    writer = csv.writer(csvFile)
    strbase = 'http://www.allitebooks.com/page/'
    imgurlbase = "img/"
    newimgurlbase='./new/'
    for i in range(1,820):
        print(i)
        strurl = strbase + str(i)
        html = requests.get(strurl)
        soup = BeautifulSoup(html.text, "lxml")
        book_list = soup.select("article")
        for book in book_list:
            title= book.select_one(".entry-title").get_text().replace("/", "_")
            url=book.select_one(".entry-title").a.get('href')
            imgurl= book.select_one('.attachment-post-thumbnail').get('src')
            author= book.select_one(".entry-author").get_text()[3:]
            print(title)
            print (author)
            d = []
            d.append(title)
            d.append(author)
            d.append(url)
            writer.writerow(d)
            title_img = imgurlbase + title + ".jpg"
            new_title_img=newimgurlbase+title+'.jpg'
            r_img = requests.get(imgurl)
            with open(title_img, 'wb') as fp:
                fp.write(r_img.content)
            img = Image.open(title_img)
            img=img.convert('RGB')
            new_img = img.resize((128, 256), Image.ANTIALIAS)
            new_img.save(new_title_img, quality=100)
    csvFile.close()
if __name__=='__main__':
    main()