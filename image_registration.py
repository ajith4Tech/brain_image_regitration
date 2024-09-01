# image_registration.py
import os
import numpy as np
import cv2
import SimpleITK as sitk
import skimage.metrics as sm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity
from image_utils.feature_extraction import extract_features


def register_image(input_path, similar_path):
    input_image = np.asarray(Image.open(input_path).convert('L'))
    similar_image = np.asarray(Image.open(similar_path).convert('L'))

    if input_image.shape != similar_image.shape:
        similar_image = cv2.resize(similar_image, (input_image.shape[1], input_image.shape[0]))

    fixed_image = sitk.GetImageFromArray(similar_image.astype(np.float32))
    moving_image = sitk.GetImageFromArray(np.asarray(Image.fromarray(input_image).convert('L')).astype(np.float32))

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=1000)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler2DTransform())
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])

    final_transform = registration_method.Execute(fixed_image, moving_image)

    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    registered_array = sitk.GetArrayFromImage(resampled_image)
    Image.fromarray(registered_array.astype(np.uint8)).save("static/images/registered_image.png")

    diff_image = sitk.Subtract(fixed_image, resampled_image)
    diff_array = sitk.GetArrayFromImage(diff_image)
    diff_array = diff_array.reshape(fixed_image.GetSize()[1], fixed_image.GetSize()[0])
    Image.fromarray(diff_array.astype(np.uint8)).save("static/images/difference_image.png")

    input_image_path = os.path.join('static', 'images', 'input_image.png')
    similar_image_path = os.path.join('static', 'images', 'similar_image.png')

    Image.fromarray(input_image).save(input_image_path)
    Image.fromarray(similar_image).save(similar_image_path)

    input_features = extract_features(input_path)
    registered_features = extract_features("static/images/registered_image.png")
    similarity = cosine_similarity(input_features.reshape(1, -1), registered_features.reshape(1, -1))
    similarity = similarity * 100

    input_image_gray = cv2.imread("static/images/input_image.png", cv2.IMREAD_GRAYSCALE)
    registered_image_gray = cv2.imread("static/images/registered_image.png", cv2.IMREAD_GRAYSCALE)

    mse = np.mean((registered_image_gray - input_image_gray) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    # Specify the data_range parameter for SSIM
    data_range = registered_array.max() - registered_array.min()
    ssim = sm.structural_similarity(input_image.astype(np.float32), registered_array.astype(np.float32), multichannel=False, data_range=data_range)
    ssim = ssim * 100

    print("Similarity: ", similarity)
    print("MSE: ", mse)
    print("PSNR: ", psnr)
    print("SSIM: ", ssim)

    result_dict = {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'similarity': similarity
    }

    return result_dict
