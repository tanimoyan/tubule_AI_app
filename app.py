from enum import unique
from flask import Flask
from flask import render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required
import os
import hashlib

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pytz

from model import FromModelMeasureHight

from PIL import Image



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
#セッション(状態の管理)情報を暗号化するためのもの
app.config['SECRET_KEY'] = os.urandom(24)
db = SQLAlchemy(app)

#flaskでログインする機能をまとめたLoinManagerクラスのインスタンス化
login_manager = LoginManager()
#LoinManagerと今回作成したappの紐付け
login_manager.init_app(app)

# # db.Modelはsqlalchemyを使用する際に用いるクラス
# class Post(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(50), nullable=False)
#     body = db.Column(db.String(300), nullable=False)
#     created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(pytz.timezone('Asia/Tokyo')))

#UserMixinはログイン機能を持ったクラス(DB=テーブルを持ったクラス)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), nullable=False, unique=True)
    password = db.Column(db.String(12))

#セッションに保存されているユーザーidからユーザー情報を読み込むために必要なもの
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'tanimoto' and hashlib.sha256(password.encode('utf-8')).hexdigest() == '819f03c3cc46f68995b15b627bec40b65a9269205543fdff966a09b9e12c3136':
            return redirect('/height')
        else:
            return redirect('/')
    else:
        return render_template('login.html')

@app.route("/height", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        f.save(filepath)

        #thresholdの値
        sure_fig_p = float(request.form.get('threshold'))

        # モデルのインポート
        result, predict_img = FromModelMeasureHight(filepath, sure_fig_p=sure_fig_p)
        
        contour_img_path = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"

        pil_img = Image.fromarray(predict_img)

        pil_img.save(contour_img_path)

        return render_template("index.html", filepath=filepath, result=result, contour_img_path=contour_img_path, sure_fig_p=sure_fig_p)

@app.route('/height')
@login_required #ログインしているユーザーしかアクセスできない(ログインしている前提)
def logput():
    logout_user()
    return redirect('/')