import numpy as np
import skimage.metrics as sm
import cv2
import matplotlib.pyplot as plt
import uuid
import base64
from io import BytesIO
from flask import url_for
import os

def calculate_psnr(input_path, registered_image_path):
    try:
        img1 = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(registered_image_path, cv2.IMREAD_GRAYSCALE)

        mse = np.mean((img1 - img2) ** 2)
        max_val = np.max(img1)
        psnr = 20 * np.log10(max_val / np.sqrt(mse))

        diff = img1 - img2
        diff = np.abs(diff) / np.max(diff)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(diff, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title('Difference Image')
        axs[1].hist(diff.flatten(), bins=50)
        axs[1].set_xlabel('Pixel Difference')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Pixel Differences')

        filename = str(uuid.uuid4()) + '.png'
        filepath = os.path.join('static', 'images', filename)
        plt.savefig(filepath, dpi=300)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()

        plt.clf()

        return psnr, plot_data, url_for('static', filename=f'images/{filename}')
    except Exception as e:
        print(f"Error in calculating PSNR: {e}")
        return None, None, None
