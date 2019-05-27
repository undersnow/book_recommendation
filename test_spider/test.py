import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
import csv
import codecs


def main():
    csvFile = open('data.csv', 'a', newline='', encoding='utf_8')
    fileHeader = ["title", "author", "url"]
    writer = csv.writer(csvFile)
    strbase = 'http://fx.sjtulib.superlib.net/s?size=15&isort=0&x=788_1071&pages='
    imgurlbase = "img/"
    for i in range(1, 201):
        print(i)
        strurl = strbase + str(i)
        r = requests.get(strurl)
        soup = BeautifulSoup(r.text, "html5lib")
        a = soup.find_all(class_="favorite")
        for j in a:

            logo = j.span.get('logo')
            title = j.span.get('title').replace("/", "_")
            author = j.span.get('author')
            appurl = j.span.get('appurl')
            d = []
            d.append(title)
            d.append(author)
            d.append(appurl)
            writer.writerow(d)
            title_img = imgurlbase + title + ".jpg"
            r_img = requests.get(logo)
            with open(title_img, 'wb') as fp:
                fp.write(r_img.content)
            img = Image.open(title_img)
            new_img = img.resize((128, 256), Image.ANTIALIAS)
            new_img.save(title_img, quality=100)
    csvFile.close()


if __name__ == '__main__':
    main()
