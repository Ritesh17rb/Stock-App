from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from models import create_article, get_articles, get_article_by_id, like_article, dislike_article, delete_article
import os
from werkzeug.utils import secure_filename

articles_bp = Blueprint("articles", __name__, url_prefix="/articles")

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- List all articles ---
@articles_bp.route("/")
def list_articles():
    articles = get_articles()
    return render_template("articles.html", articles=articles)


# --- Create new article ---
@articles_bp.route("/create", methods=["GET", "POST"])
def create():
    if "user_id" not in session:
        flash("Login required to post an article", "warning")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        title = request.form.get("title")
        content = request.form.get("content")
        file = request.files.get("image")

        if not title or not content:
            flash("Title and content are required", "danger")
            return render_template("create_article.html")

        image_url = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(filepath)
            image_url = "/" + filepath.replace("\\", "/")

        create_article(session["user_id"], title, content, image_url)
        flash("Article posted successfully", "success")
        return redirect(url_for("articles.list_articles"))

    return render_template("create_article.html")


# --- View single article ---
@articles_bp.route("/<article_id>")
def view(article_id):
    article = get_article_by_id(article_id)
    if not article:
        flash("Article not found", "danger")
        return redirect(url_for("articles.list_articles"))
    return render_template("view_article.html", article=article)


# --- Like an article ---
@articles_bp.route("/like/<article_id>")
def like(article_id):
    if "user_id" not in session:
        flash("Please login to like articles", "warning")
        return redirect(url_for("auth.login"))

    like_article(article_id)
    flash("Article liked!", "success")

    # Redirect back to the same page (either list or single article view)
    referrer = request.referrer
    if referrer and f"/articles/{article_id}" in referrer:
        return redirect(url_for("articles.view", article_id=article_id))
    else:
        return redirect(url_for("articles.list_articles"))


# --- Dislike an article ---
@articles_bp.route("/dislike/<article_id>")
def dislike(article_id):
    if "user_id" not in session:
        flash("Please login to dislike articles", "warning")
        return redirect(url_for("auth.login"))

    dislike_article(article_id)
    flash("Article disliked!", "info")

    # Redirect back to the same page (either list or single article view)
    referrer = request.referrer
    if referrer and f"/articles/{article_id}" in referrer:
        return redirect(url_for("articles.view", article_id=article_id))
    else:
        return redirect(url_for("articles.list_articles"))


# --- Delete an article ---
@articles_bp.route("/delete/<article_id>")
def delete(article_id):
    if "user_id" not in session:
        flash("Login required", "warning")
        return redirect(url_for("auth.login"))

    result = delete_article(article_id, session["user_id"])
    if result.deleted_count:
        flash("Article deleted successfully", "success")
    else:
        flash("You can only delete your own article", "danger")
    return redirect(url_for("articles.list_articles"))