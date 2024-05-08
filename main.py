import os
from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from skimage import io
from sklearn.cluster import MiniBatchKMeans

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_size_format(b, factor=1024, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

def compress_image(input_path, quality=50):
    try:
        # Read the original image
        img = Image.open(input_path)
        image = np.array(img)

        # Ensure image is RGB
        if image.ndim == 2:  # Grayscale to RGB
            image = np.stack((image,)*3, axis=-1)
        elif image.shape[2] == 4:  # RGBA to RGB
            image = image[..., :3]

        rows, cols, channels = image.shape
        pixels = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

        kmeans = MiniBatchKMeans(n_clusters=128, n_init=10, max_iter=200)
        kmeans.fit(pixels)

        clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
        labels = np.asarray(kmeans.labels_, dtype=np.uint8)
        labels = labels.reshape(rows, cols)
        colored = clusters[labels]

        # Save the compressed image
        compressed_img = Image.fromarray(colored)
        compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_' + os.path.basename(input_path))
        compressed_img.save(compressed_path, format="JPEG", quality=quality)

        return compressed_path

    except Exception as e:
        print(f"Error compressing image: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compress", methods=["POST"])
def compress():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files["file"]
    operation = request.form.get("operation")
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        if operation == "2":  # Clear Uploaded Image
            return redirect(url_for('index'))

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(upload_path)
        compressed_path = compress_image(upload_path)

        if compressed_path:
            original_size = os.path.getsize(upload_path)
            compressed_size = os.path.getsize(compressed_path)
            
            return render_template('result.html',
                                   original_path=filename,
                                   compressed_path=os.path.basename(compressed_path),
                                   original_size=get_size_format(original_size),
                                   compressed_size=get_size_format(compressed_size),
                                   saving_percentage=((original_size - compressed_size) / original_size) * 100
                                   )
        else:
            flash('Error compressing image!!!')
            return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
