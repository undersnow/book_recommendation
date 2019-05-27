# book_recommendation
自动识别图片并推荐相应图书

	
本项目基于三方
神经网络  网页  爬虫 
各方之间保持独立性，在自己的分支上写，最后合并

三方之间应有统一的接口函数：

神经网络：
getSimliarPhotos（查询图片路径，返回图片数量）  
以列表形式返回相应数目的图片路径

爬虫：
getPosition()
返回爬取图片的文件路径（所有爬取图片存储在统一的文件夹下）

getInformation（图片路径）返回相应信息的url


注意：
爬取图片存放地址须符合格式 
爬取图片    A\gallery\1\爬取所有图片
查询图片    A\query\1\待查询图片
getPosition()返回文件夹A的地址
				  

环境：
pytorch 0.3+
numpy