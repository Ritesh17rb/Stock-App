# utils.py
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config

ALLOWED_EXT = {'png','jpg','jpeg','gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def save_file_to_disk(file_storage, folder=None):
    folder = folder or Config.UPLOAD_FOLDER
    os.makedirs(folder, exist_ok=True)
    filename = secure_filename(file_storage.filename)
    path = os.path.join(folder, filename)
    file_storage.save(path)
    return path

def hash_password(password):
    return generate_password_hash(password)

def verify_password(hashval, password):
    return check_password_hash(hashval, password)
