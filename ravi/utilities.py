import numpy as np
from skimage.metrics import structural_similarity


def create_array_with_zeros(n, hole_ratio, number_of_holes=1):
    # Create an array of ones
    arr = np.ones((n, n))

    # Calculate the size of the square containing zeros
    zero_size = n // hole_ratio

    for _ in range(number_of_holes):
        # Generate random coordinates for the top-left corner of the zero square
        start_row = np.random.randint(0, n - zero_size + 1)
        start_col = np.random.randint(0, n - zero_size + 1)

        # Set the values in the zero square region to zeros
        arr[start_row : start_row + zero_size, start_col : start_col + zero_size] = 0

    return arr


def create_strip_mask(n, thickness, orientation=None, number=1):
    # Create an array of ones
    arr = np.ones((n, n))
    for _ in range(number):
        if orientation == "H":
            start_row = np.random.randint(0, n - thickness + 1)
            arr[start_row : start_row + thickness, :] = 0
        elif orientation == "V":
            start_col = np.random.randint(0, n - thickness + 1)
            arr[:, start_col : start_col + thickness] = 0
        else:
            coin_flip = bool(np.random.randint(0, 2))
            if coin_flip:
                start_row = np.random.randint(0, n - thickness + 1)
                arr[start_row : start_row + thickness, :] = 0
            else:
                start_col = np.random.randint(0, n - thickness + 1)
                arr[:, start_col : start_col + thickness] = 0
    return arr


def calculate_ssim(ground_truth, prediction, mask):
    # Compute SSIM between two images only in the masked region
    # masked_ground_truth = ground_truth * mask
    # score_baseline = structural_similarity(masked_ground_truth, ground_truth,  data_range=np.max(ground_truth) - np.min(ground_truth))
    ground_truth = ground_truth[np.where(mask == 0)]
    prediction = prediction[np.where(mask == 0)]
    score = structural_similarity(
        ground_truth, prediction, data_range=np.max(prediction) - np.min(prediction)
    )
    # normalized_score = (score-score_baseline)/(1-score_baseline)
    return score


def calculate_psnr(ground_truth, prediction, mask):
    # Compute PSNR between two images only in the masked region
    ground_truth = ground_truth[np.where(mask == 0)]
    prediction = prediction[np.where(mask == 0)]
    mse = np.mean((ground_truth - prediction) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(prediction)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_rmse(ground_truth, prediction, mask):
    # Compute RMSE between two images only in the masked region
    ground_truth = ground_truth[np.where(mask == 0)]
    prediction = prediction[np.where(mask == 0)]
    rmse = np.sqrt(np.mean((ground_truth - prediction) ** 2))
    return rmse
