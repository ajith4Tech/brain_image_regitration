# brainmlem.py
import os
import numpy as np
import cv2
import skimage.metrics as sm
from sklearn.metrics.pairwise import cosine_similarity
from image_utils.feature_extraction import extract_features

def align_images(input_path, similar_path):
    try:
        # Read images
        input_image = cv2.imread(input_path)
        if input_image is None:
            raise ValueError(f"Input image not found: {input_path}")

        similar_image = cv2.imread(similar_path)
        if similar_image is None:
            raise ValueError(f"Similar image not found: {similar_path}")

        # Ensure images are resized to the same dimensions
        target_size = (256, 256)
        input_image = cv2.resize(input_image, target_size)
        similar_image = cv2.resize(similar_image, target_size)

        # Feature detection and matching
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints1, descriptors1 = orb.detectAndCompute(input_image, None)
        keypoints2, descriptors2 = orb.detectAndCompute(similar_image, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        distances = [match.distance for match in matches]
        variance = np.mean(distances)
        distance_threshold = 1.5 * variance
        matches = [match for match in matches if match.distance < distance_threshold]

        # Draw matched keypoints
        matched_image = cv2.drawMatches(input_image, keypoints1, similar_image, keypoints2, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Find homography and align images
        source_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        destination_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
        transform_matrix, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)

        aligned_image = cv2.warpPerspective(input_image, transform_matrix, target_size)

        # Convert to grayscale for comparison
        aligned_image_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Save images for debugging or later use
        folder_path = "static/images/"
        cv2.imwrite(folder_path + "input_image.png", input_image)
        cv2.imwrite(folder_path + "similar_image.png", similar_image)
        cv2.imwrite(folder_path + "matched_image.png", matched_image)
        cv2.imwrite(folder_path + "aligned_image.png", aligned_image)

        # Compute metrics
        mse = np.mean((aligned_image_gray - input_image_gray) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        ssim = sm.structural_similarity(input_image_gray, aligned_image_gray)

        # Compute similarity score
        similarity_score = cosine_similarity(extract_features(input_path).reshape(1, -1),
                                             extract_features(folder_path + "aligned_image.png").reshape(1, -1))

        print(f'MSE: {mse}')
        print(f'PSNR: {psnr}')
        print(f'SSIM: {ssim * 100}')
        print(f'Similarity score: {similarity_score}')

        results = {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'similarity': similarity_score,
            'num_matches_before': len(matches),
            'num_matches_after': len([match for match in matches if match.distance < distance_threshold])
        }

        return results
    except Exception as e:
        print(f"Error in aligning images: {e}")
        return None
