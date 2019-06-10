'''
    姓名:李开涞
    文件描述:网站模块
'''
import os
import random
from flask import Flask, flash, request, redirect, url_for, render_template, g, send_from_directory
from werkzeug.utils import secure_filename
import sqlite3
import time
import shutil


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    #设置用户上传的文件夹名，以及数据库中所有包含的图片
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploaded')
    MEDIA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
    #限制允许上传的文件扩展名
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MEDIA_FOLDER'] = MEDIA_FOLDER

    # 初始化数据库  
    from . import db
    db.init_app(app)


    # 通过扩展名判断文件是否允许上传
    def allowed_file(filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    # 从神经网络得到结果，再经由数据库，返回相似文件的文件名
    def find_sim_files(file_folder):
        from . import neural_network
        sim_image_paths = neural_network.chooseImage.getSimliarPhotos(MEDIA_FOLDER, file_folder, 10)
        sim_files = []
        for sim_image_path in sim_image_paths:
            name_jpg = os.path.basename(sim_image_path)
            name = os.path.splitext(name_jpg)[0]
            try:
                book = db.query_db(f"select * from book where name = '{name}'")
            except:
                book = [{'URL':''}]
            img_path = url_for('get_image', filename=name_jpg)
            sim_files.append((name, img_path, book[0]['URL']))

        return sim_files


    # 主页面，处理用户上传
    @app.route('/', methods=['GET', 'POST'])
    def upload_file(error=None):
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return render_template('main.html', error='No selected file')
            if file and allowed_file(file.filename):
                file_id = str(random.randint(0, 100000))
                # 重命名文件
                filename = file_id + '.' + file.filename.split('.')[-1]
                timestamp = str(time.time_ns())
                # 生成临时文件夹
                file_folder = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
                os.makedirs(os.path.join(file_folder, '1'))
                # 保存文件到临时文件夹
                file.save(os.path.join(file_folder, '1', filename))
                return redirect(url_for('result',
                                        filename=filename, timestamp=timestamp))
            elif file and not allowed_file(file.filename):
                return render_template('main.html', error='File type not allowed')
        return render_template('main.html')


    # 结果显示页面
    @app.route('/uploads/<timestamp>/<filename>')
    def result(filename, timestamp):
        file_folder = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
        # file_path = os.path.join(file_folder, '1', filename)
        try:
            sim_files = find_sim_files(file_folder)
        except:
            sim_files = []
        try:
            shutil.rmtree(file_folder)
        except:
            pass
        return render_template('result.html', sim_files = sim_files)

    # 图片显示
    @app.route('/media/<filename>')
    def get_image(filename):
        return send_from_directory(os.path.join(app.config['MEDIA_FOLDER'], '1'),
                               filename)

    return app
