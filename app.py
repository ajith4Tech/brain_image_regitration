from flask import Flask, render_template, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
from image_utils.metrics import calculate_psnr
from image_utils.image_similarity import image_classification
import brainregistration
from image_registration import register_image
import os
import uuid
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key')  # For flash messages

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_file(file):
    """Save the uploaded file to the server."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
    return None

def plot_histogram(dists):
    """Plot histogram of distances and save as a PNG file."""
    plt.hist(dists.flatten(), bins=50)
    plt.title('Histogram of Manhattan Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')

    filename = str(uuid.uuid4()) + '.png'
    filepath = os.path.join('static', 'images', filename)
    plt.savefig(filepath, dpi=300)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()

    plt.clf()
    return plot_data, url_for('static', filename=f'images/{filename}')

def process_results(filepath, classify_path, dists, method):
    """Process results based on the selected method."""
    logging.info(f"Processing results for method: {method}")
    try:
        if method == 'Intensity-Based':
            logging.info("Using Intensity-Based registration")
            results = register_image(filepath, classify_path)
            if results is None:
                raise ValueError("Registration failed. No results returned.")
            psnr_func = calculate_psnr
        else:
            logging.info("Using Feature-Based registration")
            results = brainregistration.align_images(filepath, classify_path)
            if results is None:
                raise ValueError("Alignment failed. No results returned.")
            psnr_func = calculate_psnr

        # Assuming plot_histogram returns a tuple (plot_data, plot_url)
        plot_data, plot_url = plot_histogram(dists)

        # Safely update the results dictionary
        if isinstance(results, dict):
            results.update({'plot_data': plot_data, 'plot_url': plot_url})
        else:
            raise ValueError("Results object is not a dictionary.")

        # Calculate PSNR and handle the result
        psnr, plot_psnr, image_url = psnr_func(
            'static/images/input_image.png', 
            'static/images/registered_image.png' if method == 'Intensity-Based' else 'static/images/aligned_image.png'
        )
        
        logging.info(f"Processed results successfully for method: {method}")
        return results, image_url

    except Exception as e:
        logging.error(f"Error processing results: {e}")
        flash('An error occurred while processing the results.')
        return {}, None



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_registration', methods=['POST', 'GET'])
def image_registration():
    if 'image' not in request.files or request.files['image'].filename == '':
        flash('No file uploaded')
        return redirect(request.url)

    file = request.files['image']
    filepath = save_file(file)
    
    if not filepath:
        return redirect(request.url)

    submit_type = request.form.get('submit_type')
    if submit_type not in ['Intensity-Based', 'Feature-Based']:
        flash('Invalid submit type')
        return redirect(request.url)

    try:
        classify_path, dists = (
            image_classification(filepath)
        )
        
        results, image_url = process_results(filepath, classify_path, dists, submit_type)

        return render_template(
            'intensity_based_result.html' if submit_type == 'Intensity-Based' else 'feature_based_result.html', 
            results=results, 
            image_url=image_url
        )
    except Exception as e:
        logging.error(f"Error during registration: {e}")
        flash('An error occurred during image processing.')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
