from enum import unique
from flask import Flask
from flask import render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required
import os

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

#db.Modelはsqlalchemyを使用する際に用いるクラス
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

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        #長さが256ビットの適当な値を返す
        user = User(username=username, password=generate_password_hash(password, method='sha256'))

        #データベースに追加
        db.session.add(user)
        #データベースに保存->変更が保存反映
        db.session.commit()
        return redirect('/login')
    else:
        return render_template('signup.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        #usernameでフィルターをかけてそこに合致したものを取ってくる(Userのテーブルから取ってくる)
        user = User.query.filter_by(username=username).first() #複数取らないように始めだけ取るfirst()
        #userのパスワード情報とフォームに入力したパスワード情報が等しいか判別する
        if check_password_hash(user.password, password):
            login_user(user) #ユーザー名が等しければログインし、ホームページに飛ぶ
            return redirect('/')
    else:
        return render_template('login.html')

@app.route('/logout')
@login_required #ログインしているユーザーしかアクセスできない(ログインしている前提)
def logput():
    logout_user()
    return redirect('/login')

@app.route("/", methods=["GET", "POST"])
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

        # # 予測を実施
        # output = model(image)
        # _, prediction = torch.max(output, 1)
        # result = prediction[0].item()

        return render_template("index.html", filepath=filepath, result=result, contour_img_path=contour_img_path, sure_fig_p=sure_fig_p)