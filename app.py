# app.py
from flask import Flask, render_template
from config import Config
from blueprints.auth import auth
from blueprints.profile import profile
from blueprints.stocks import stocks
import os

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)
    app.secret_key = Config.SECRET_KEY

    app.register_blueprint(auth)
    app.register_blueprint(profile)
    app.register_blueprint(stocks)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
