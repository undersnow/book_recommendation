import os
import random
from flask import Flask, flash, request, redirect, url_for, render_template, g, send_from_directory
from werkzeug.utils import secure_filename
import sqlite3


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

    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploaded')
    MEDIA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MEDIA_FOLDER'] = MEDIA_FOLDER

    from . import db
    db.init_app(app)


    def allowed_file(filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    def find_sim_files(file_path):
        from . import neural_network
        sim_image_paths = neural_network.chooseImage.getSimliarPhotos(MEDIA_FOLDER, UPLOAD_FOLDER, 10)
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
                filename = file_id + '.' + file.filename.split('.')[-1]
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], '1', filename))
                return redirect(url_for('result',
                                        filename=filename))
            elif file and not allowed_file(file.filename):
                return render_template('main.html', error='File type not allowed')
        return render_template('main.html')


    @app.route('/uploads/<filename>')
    def result(filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], '1', filename)
        sim_files = find_sim_files(file_path)
        try:
            os.remove(file_path)
        except:
            pass
        return render_template('result.html', sim_files = sim_files)

    @app.route('/media/<filename>')
    def get_image(filename):
        return send_from_directory(os.path.join(app.config['MEDIA_FOLDER'], '1'),
                               filename)

    return app
