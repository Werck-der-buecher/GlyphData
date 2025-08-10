import pathlib
import random
import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
from glyphdata.glyph.glyph import Glyph
from glyphdata.utils.io_utils import decompress_pickle
from perlin_noise import PerlinNoise
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, lognorm
from skimage import measure
from skimage.transform import rescale
from skimage.util import img_as_float

warnings.filterwarnings("ignore")

# random seed initialization
np.random.seed(0)
random.seed(0)

# settings
INK_DIFFUSE_COUNT_FACTOR = 6
INK_DIFFUSE_SIZE_RATIO = 12
PROB_RESIDUAL = 0.8
MARGIN_FACTOR = 0.12

# augmentations
aug_morpho = A.OneOf([A.Morphological(p=1, scale=(2, 5), operation='dilation'),
                      A.Morphological(p=1, scale=(2, 3), operation='erosion'), ], p=0.8)
aug_rotate = A.SafeRotate(limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0)
aug_course_dropout = A.CoarseDropout(
    fill_value=1,
    num_holes_range=(1, 8),
    hole_height_range=(4, 32),
    hole_width_range=(4, 32),
    p=0.5)
aug_fine_dropout = A.CoarseDropout(
    fill_value=1,
    num_holes_range=(10, 120),
    hole_height_range=(1, 4),
    hole_width_range=(1, 4),
    p=0.5)
aug_noise = A.CoarseDropout(
    fill_value='random',
    num_holes_range=(1, 20),
    hole_height_range=(1, 3),
    hole_width_range=(1, 3),
    p=0.5)
aug_elastic = A.ElasticTransform(
    alpha=10,
    sigma=5,
    interpolation=1,
    border_mode=4,
    p=1.0,
)


# contour analysis
def compute_curvature(point, i, contour, window_size):
    start = max(0, i - window_size // 2)
    end = min(len(contour), i + window_size // 2 + 1)
    neighborhood = contour[start:end]
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original) // 2])
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    coeffs = np.polyfit(rotated_x, rotated_y, 2)
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    return np.mean(curvature)


def compute_curvature_profile(mask, min_contour_length, window_size_ratio):
    contours = measure.find_contours(mask)

    curvature_values = []
    edge_pixels = []

    for contour in contours:
        for i, point in enumerate(contour):
            if contour.shape[0] > min_contour_length:
                window_size = int(contour.shape[0] / window_size_ratio)
                curvature = compute_curvature(point, i, contour, window_size)
                curvature = curvature
                curvature_values.append(curvature)
                edge_pixels.append(point)

    curvature_values = np.array(curvature_values)
    edge_pixels = np.array(edge_pixels)

    return edge_pixels, curvature_values


def plot_edges_with_curvature(mask, min_contour_length, window_size_ratio):
    edge_pixels, curvature_values = compute_curvature_profile(mask, min_contour_length, window_size_ratio)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mask, cmap='gray_r')
    ax[1].imshow(mask, cmap='gray_r')

    threshold = np.percentile(np.abs(curvature_values), 90)
    scat = ax[0].scatter(edge_pixels[:, 1], edge_pixels[:, 0], c=curvature_values, cmap='coolwarm', s=8,
                         vmin=-threshold, vmax=threshold)
    plt.colorbar(scat, label='Curvature', ax=ax[0])
    scat = ax[1].scatter(edge_pixels[:, 1], edge_pixels[:, 0], c=abs(curvature_values), cmap='coolwarm', s=8, vmin=0,
                         vmax=threshold)
    plt.colorbar(scat, label='Curvature', ax=ax[1])

    plt.title("Curvature of Binary Mask")
    plt.show()


def approx_ink_diffusion(mask: np.ndarray,
                         min_contour_length,
                         window_size_ratio,
                         curvature_q: int = 80,
                         diff_speed: float = 0.1,
                         num_diff_steps: int = 50,
                         include_convex: bool = False):
    edge_pixels, curvature_values = compute_curvature_profile(mask, min_contour_length, window_size_ratio)

    # Initialize the ink diffusion map
    diffusion_grid = np.zeros_like(mask, dtype=np.float32)

    if len(curvature_values) == 0:
        return diffusion_grid

    # Initial ink distribution based on curvature
    for i, point in enumerate(edge_pixels):
        x, y = point
        c = -curvature_values[i]
        if include_convex or c > 0:
            diffusion_grid[round(x), round(y)] = abs(c)

    if include_convex:
        perc = np.percentile(abs(curvature_values), q=curvature_q)
    else:
        perc = np.percentile(curvature_values[curvature_values > 0], q=curvature_q)
    diffusion_grid = np.clip(diffusion_grid, a_min=0, a_max=perc)

    # heat diffusion
    for _ in range(num_diff_steps):
        diffusion_grid = diffusion_grid + diff_speed * scipy.ndimage.laplace(diffusion_grid)

    diffusion_grid = cv2.normalize(diffusion_grid, None, 0, 1, cv2.NORM_MINMAX)

    return diffusion_grid


def augment_glyph(char: str,
                  unicode: int,
                  img: np.ndarray,
                  residuals: list[np.ndarray],
                  num_augmentations: int,
                  image_size: int
                  ) -> tuple[list[np.ndarray]]:
    min_contour_length = image_size // INK_DIFFUSE_COUNT_FACTOR
    window_size_ratio = image_size // INK_DIFFUSE_SIZE_RATIO

    char_images = []
    residual_images = []
    composite_images = []
    target_names = []
    for augmentation_idx in range(num_augmentations):
        try:
            # (0) Erode/dilate character
            img_morpho = aug_morpho(image=img)['image']

            # Extract bbox around character
            x_min, y_min, x_max, y_max = A.functional.bbox_from_mask(img_morpho)
            char_img = img_morpho[y_min:y_max, x_min:x_max]

            # (1) Rotate character
            img_rot = aug_rotate(image=char_img)['image']

            # (2) Scale character
            H, W = img_rot.shape
            diff = image_size - max(H, W)
            sf_upper = image_size / max(H, W)
            sf_lower = 1 - (sf_upper - 1) / 2

            if sf_upper != sf_lower:
                sf_nbins = 20
                sf_bins = np.linspace(sf_lower, sf_upper, sf_nbins)
                sf_mu = 1.0
                sf_sigma = (sf_upper - sf_lower) / 4
                sf_scale = np.exp(sf_sigma ** 2)  # scale to ensure that mode is at 1
                sf_probs = lognorm.pdf(sf_bins, s=sf_sigma, scale=sf_scale)
                sf_probs /= sf_probs.sum()
                sampled_sf = np.random.choice(sf_bins, p=sf_probs)
                img_rescale = rescale(img_rot, sampled_sf, anti_aliasing=False)
            else:
                img_rescale = img_rot

            # (3) Shift & paste image onto canvas
            H, W = img_rescale.shape

            # y-pos normal distribution sample probs
            y_nbins = image_size - H + 1
            y_bins = np.arange(y_nbins)
            y_mu = (y_nbins - 1) / 2
            y_sigma = y_nbins / 4
            y_probs = norm.pdf(y_bins, y_mu, y_sigma)
            y_probs /= y_probs.sum()
            # x-pos normal distribution sample probs
            x_nbins = image_size - W + 1
            x_bins = np.arange(x_nbins)
            x_mu = (x_nbins - 1) / 2
            x_sigma = x_nbins / 4
            x_probs = norm.pdf(x_bins, x_mu, x_sigma)
            x_probs /= x_probs.sum()

            sampled_y_pos = np.random.choice(y_bins, p=y_probs)
            sampled_x_pos = np.random.choice(x_bins, p=x_probs)

            img_canvas = np.zeros((image_size, image_size))
            img_canvas[sampled_y_pos:(sampled_y_pos + H), sampled_x_pos:(sampled_x_pos + W)] = img_rescale

            # (4) Paste character residuals to image borders
            gaussian_sigma = np.random.choice(np.arange(0, 4), p=[0.55, 0.25, 0.15, 0.05])
            residual_canvas = np.zeros((image_size, image_size))

            if np.random.random() < PROB_RESIDUAL:
                margin_area = np.ones((image_size, image_size))
                margin_area[int(image_size * MARGIN_FACTOR):image_size - int(image_size * MARGIN_FACTOR),
                int(image_size * MARGIN_FACTOR):image_size - int(image_size * MARGIN_FACTOR)] = 0

                # additionally remove char area from mask
                extra_space = int(0.1 * image_size)
                slice_h = slice(max(0, sampled_y_pos - extra_space), min(image_size, sampled_y_pos + H + extra_space))
                slice_w = slice(max(0, sampled_x_pos - extra_space), min(image_size, sampled_x_pos + W + extra_space))
                margin_area[slice_h, slice_w] = 0

                margin_coords = np.c_[np.nonzero(margin_area)]

                if len(margin_coords):
                    residual_canvas = np.zeros((image_size, image_size))
                    num_residuals = np.random.randint(low=1, high=5)
                    for _ in range(num_residuals):
                        margin_idx = np.random.choice(len(margin_coords))
                        m_y, m_x = margin_coords[margin_idx]

                        res_idx = np.random.choice(len(residuals))
                        res_img = residuals[res_idx]
                        x_min, y_min, x_max, y_max = A.functional.bbox_from_mask(res_img)
                        res_img = res_img[y_min:y_max, x_min:x_max]
                        res_img = A.LongestMaxSize(max(H, W), interpolation=cv2.INTER_LINEAR)(image=res_img)['image']
                        res_img = img_as_float(res_img)

                        paste_H = min((image_size - m_y), res_img.shape[0])
                        paste_W = min((image_size - m_x), res_img.shape[1])
                        residual_canvas[m_y:m_y + paste_H, m_x:m_x + paste_W] = res_img[:paste_H, :paste_W]

                    # residual_canvas = np.logical_and(residual_canvas, margin_area)
                    residual_canvas = np.where(margin_area, residual_canvas, margin_area)
                    residual_canvas = gaussian_filter(residual_canvas, sigma=gaussian_sigma)

            composite_img = gaussian_filter(img_canvas + residual_canvas, sigma=gaussian_sigma)

            # (5) Simulate ink diffusion in concave regions
            if np.random.random() > 0.5:
                diff_steps = np.random.randint(0, 80)
                diff_factor = np.random.uniform(1, 2)
                if diff_steps > 0:
                    diffusion_grid = approx_ink_diffusion(composite_img, min_contour_length, window_size_ratio,
                                                          curvature_q=80, num_diff_steps=diff_steps,
                                                          include_convex=True)
                    _, ink_mask = cv2.threshold(diffusion_grid, 0.01, 1.0, cv2.THRESH_BINARY)
                    cutoff_area = gaussian_filter(
                        cv2.bitwise_or((composite_img > 0.5).astype(int), ink_mask.astype(int)).astype(np.float32), 0.5)
                    composite_img = np.clip(
                        np.where(cutoff_area, composite_img + diff_factor * diffusion_grid, composite_img), 0, 1)

            # (6) Apply elastic deformation
            composite_img = aug_elastic(image=composite_img)['image']

            # (7) Apply global Perlin noise
            pnoise = PerlinNoise(octaves=1, seed=np.random.randint(0, 1000000))
            noise_map = np.array(
                [[pnoise([i / image_size, j / image_size]) for j in range(image_size)] for i in range(image_size)])
            alpha = np.random.uniform(low=0.0, high=0.4)

            composite_img = composite_img + alpha * noise_map

            # (8) Generate dropout mask to reduce intensity in certain regions
            noise_intensity = np.random.random()
            dropout_coarse = np.zeros((image_size, image_size))
            dropout_coarse = aug_course_dropout(image=dropout_coarse)['image']
            dropout_coarse = gaussian_filter(dropout_coarse, sigma=np.random.choice(np.arange(2, 4)))
            dropout_coarse *= noise_intensity

            dropout_fine = np.zeros((image_size, image_size))
            dropout_fine = aug_fine_dropout(image=dropout_fine)['image']
            dropout_fine = gaussian_filter(dropout_fine, sigma=np.random.choice(np.arange(0, 2)))
            dropout_fine *= noise_intensity

            composite_img = composite_img * (1 - np.maximum(dropout_coarse, dropout_fine))

            # (9) Random global intensity
            glob_intensity = np.random.uniform(low=0.75, high=1.0)
            composite_img = glob_intensity * composite_img

            # (10) Splatter background noise
            composite_img = aug_noise(image=composite_img)['image']

            # (11) Clip to range [0,1]
            composite_img = np.clip(composite_img, a_min=0, a_max=1)

            char_images.append(img_canvas)
            residual_images.append(residual_canvas)
            composite_images.append(composite_img)
            target_names.append(f"{char}_{unicode}_augidx={augmentation_idx}")

        except Exception as e:
            print("Error occured during processing. Skipping glyph...", e)
            continue

    return char_images, residual_images, composite_images, target_names
