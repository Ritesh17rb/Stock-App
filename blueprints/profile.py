# blueprints/profile.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from models import find_user_by_id, update_user, delete_user
from utils import allowed_file, save_file_to_disk, hash_password
from functools import wraps

profile = Blueprint("profile", __name__, url_prefix="")

def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        if "user_id" not in session:
            flash("Please login first", "warning")
            return redirect(url_for("auth.login"))
        return fn(*a, **k)
    return wrapper

@profile.route("/profile")
@login_required
def view_profile():
    user = find_user_by_id(session["user_id"])
    return render_template("profile.html", user=user)

@profile.route("/profile/edit", methods=["POST"])
@login_required
def edit_profile():
    user_id = session["user_id"]
    update_doc = {}
    if request.form.get("name"):
        update_doc["name"] = request.form.get("name")
    if request.form.get("meta") is not None:
        update_doc["meta"] = request.form.get("meta")
    avatar = request.files.get("avatar")
    if avatar and allowed_file(avatar.filename):
        update_doc["avatar"] = save_file_to_disk(avatar)
    if request.form.get("password"):
        update_doc["password"] = hash_password(request.form.get("password"))
    if update_doc:
        update_user(user_id, update_doc)
        flash("Profile updated", "success")
    else:
        flash("No changes submitted", "info")
    return redirect(url_for("profile.view_profile"))

@profile.route("/profile/delete", methods=["POST"])
@login_required
def delete_account():
    user = find_user_by_id(session["user_id"])
    if user:
        delete_user(str(user["_id"]))
    session.pop("user_id", None)
    flash("Account deleted", "info")
    return redirect(url_for("index"))
