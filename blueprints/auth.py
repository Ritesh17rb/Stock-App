# blueprints/auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import create_user, find_user_by_email
from utils import hash_password, verify_password, save_file_to_disk, allowed_file

auth = Blueprint("auth", __name__, url_prefix="")

@auth.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    if not (name and email and password):
        flash("Name, email and password required", "danger")
        return redirect(url_for("auth.signup"))
    if find_user_by_email(email):
        flash("Email already registered", "danger")
        return redirect(url_for("auth.signup"))

    avatar_path = None
    avatar = request.files.get("avatar")
    if avatar and allowed_file(avatar.filename):
        avatar_path = save_file_to_disk(avatar)

    create_user({
        "name": name,
        "email": email,
        "password": hash_password(password),
        "avatar": avatar_path,
        "meta": request.form.get("meta", "")
    })
    flash("Account created, please login", "success")
    return redirect(url_for("auth.login"))

@auth.route("/login", methods=["GET","POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    email = request.form.get("email")
    password = request.form.get("password")
    user = find_user_by_email(email)
    if not user or not verify_password(user["password"], password):
        flash("Invalid credentials", "danger")
        return redirect(url_for("auth.login"))
    session["user_id"] = str(user["_id"])
    flash("Logged in", "success")
    return redirect(url_for("stocks.dashboard"))

@auth.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out", "info")
    return redirect(url_for("index"))
