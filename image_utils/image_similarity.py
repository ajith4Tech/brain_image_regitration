# classification.py
import os
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, cosine_similarity
from image_utils.feature_extraction import extract_features

def image_classification(input_path):
    dataset_path = "Dataset/images/sample"
    print(input_path)
    try:
        input_features = extract_features(input_path)
        if input_features is None:
            return None, None

        dataset_features = []
        print(dataset_features)
        for file in os.listdir(dataset_path):
            if file.endswith(".png"):
                file_path = os.path.join(dataset_path, file)
                features = extract_features(file_path)
                if features is not None:
                    dataset_features.append(features)

        dataset_features = np.array(dataset_features)
        dists = manhattan_distances(input_features.reshape(1, -1), dataset_features)
        index = np.argmin(dists)
        similarities = cosine_similarity(input_features.reshape(1, -1), dataset_features)

        similar_path = os.path.join(dataset_path, os.listdir(dataset_path)[index])
        similar = (similarities[0][index]) * 100
        print(f"Similarity score: {similar}")

        return similar_path, dists
    except Exception as e:
        print(f"Error in image classification: {e}")
        return None, None
