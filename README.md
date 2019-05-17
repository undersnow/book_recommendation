# book_recommendation
自动识别图片并推荐相应图书


爬虫代码放在crawler文件夹中
	爬虫应爬取 书的封面的图片 以及与这本书相关的某种结果 
	并且这种结果要求可以通过图片得到
	
	爬取图片放在images文件夹中
	
神经网络相关代码放在neural network文件夹中

需要用户查询的图片（书的封面）放在query_images文件夹中

主文件main.py
	若查询图片为 image1.jpg
	运行main.py  --image image1.jpg 后会自动生成查询结果（例如返回images文件夹中与image1.jpg最相似的10张图片）
    并允许用户通过这些图片获取与这本书相关的某种结果，例如下载链接，详细信息页面等等
	
	
