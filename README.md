# 基于神经网络的图书查询
该项目会自动返回与输入图书封面最相近的图书



首先请按以下步骤进行环境的配置

### 配置

###### 本项目使用python3.7，基本库信息如下

- torch >= 1.1.0
- torchvision >= 0.3.0
- numpy >= 1.16.3
- Flask >= 1.0.3

###### 推荐使用python虚拟环境

```
$ mkdir your_dir
$ cd your_dir
$ python3.7 -m venv venv
$ source venv/bin/activate
```

###### 爬虫模块相对独立，所需包如下：

```
pip install csv
pip install lxml
pip install requests
pip install bs4
pip install pandas
pip install PIL
pip install codecs
```



###### 神经网络和网站部署的相关包，通过以下命令安装

```
$ pip install -r requirements.txt

```

###### 配置服务器相关，执行命令

```
$ export FLASK_APP=book_recommendation
$ export FLASK_ENV=development
$ flask run
```



###### 本项目使用了Market1501数据集，下载链接：

http://www.liangzheng.com.cn/Project/project_reid.html



## 开始

###### 网络训练

运行 train.py （位于目录book_recommendation\neural_network）训练网络，需根据实际需要，更改train.py中相应路径（需配合下载的数据集路径），运行完毕后会生成net.pth文件，该文件保存了训练好的网络模型，预计用时2小时



###### 图片爬取

可根据需要，运行shuxianglib.py，asynicospider.py（位于目录test_spider）前者为计算机专业书籍及交大图书馆书籍，后者为大众生活类书籍网站书籍



###### 计算特征值

运行choose.py（位于目录book_recommendation\neural_network），该程序会计算爬取的图片的特征值并生成mat文件，注意：加载爬取的图片时,程序中的路径需要根据实际修改



###### 然后根据部署文档进行网站部署即可

注意：部署时mat文件和pth文件均需放在book_recommendation\neural_network目录下

部署文档：[部署](部署文档.pdf)



## 展示

展示视频 [demo](demo/demo.mp4)



## 部署参考手册

请参见 [flask app deployment](http://flask.pocoo.org/docs/1.0/tutorial/deploy/)