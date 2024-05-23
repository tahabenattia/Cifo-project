from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error

def evaluate_image_quality(target_image, generated_image):
    # Calculate SSIM
    ssim_score = ssim(target_image, generated_image, data_range=generated_image.max() - generated_image.min())

    # Calculate PSNR
    psnr_score = psnr(target_image, generated_image, data_range=generated_image.max() - generated_image.min())

    # Calculate MSE
    mse_score = mean_squared_error(target_image, generated_image)

    return ssim_score, psnr_score, mse_score