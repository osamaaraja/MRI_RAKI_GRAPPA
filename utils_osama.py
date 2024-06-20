import numpy as np
import scipy
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim


def fft_centered(
    input: np.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    dim: Optional[Tuple[int, ...]] = None,
    norm: Optional[str] = None
) -> np.ndarray:

    input = np.fft.ifftshift(input, axes=dim)
    input = np.fft.fftn(input, s=shape, axes=dim, norm=norm)
    input = np.fft.fftshift(input, axes=dim)
    return input


def ifft_centered(
    input: np.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    dim: Optional[Tuple[int, ...]] = None,
    norm: Optional[str] = None
) -> np.ndarray:

    input = np.fft.ifftshift(input, axes=dim)
    input = np.fft.ifftn(input, s=shape, axes=dim, norm=norm)
    input = np.fft.fftshift(input, axes=dim)
    return input
def root_sum_squares(
    input: np.ndarray,
    dim: int,
    complex: Optional[bool] = None
) -> np.ndarray:

    if complex:
        input = np.sum(np.square(input), axis=-1)
    else:
        input = np.square(np.abs(input))
    return np.sqrt(np.sum(input, axis=dim))


def undersample(kgt, auto_calib_lines, R):

    start_line = (kgt.shape[0] - auto_calib_lines) // 2
    end_line = start_line + auto_calib_lines

    mask = np.zeros_like(kgt)
    mask[start_line:end_line, :] = 1
    mask[::R, :] = 1

    undersampled_data = kgt * mask

    plt.imshow(np.abs(undersampled_data[:,:,0]), cmap='gray', norm=colors.PowerNorm(gamma=0.2))
    plt.title(f'Undersampled k-space x{R}')
    plt.show()

    return undersampled_data


def only_ACS(kgt, auto_calib_lines):
    start_line = (kgt.shape[0] - auto_calib_lines) // 2
    end_line = start_line + auto_calib_lines
    mask = np.zeros_like(kgt)
    mask[start_line:end_line, :] = 1
    ACS = kgt * mask

    plt.imshow(np.abs(ACS[:,:,0]), cmap='gray', norm=colors.PowerNorm(gamma=0.2))
    plt.title(f'ACS = {auto_calib_lines}')
    plt.show()

    return ACS

def only_ACS_1(kgt, auto_calib_lines):
    start_line = (kgt.shape[0] - auto_calib_lines) // 2
    end_line = start_line + auto_calib_lines

    central_region = kgt[start_line:end_line, :]

    mask = np.zeros_like(central_region)
    mask[:, :] = 1
    ACS = central_region * mask

    plt.imshow(np.abs(ACS[:,:,0]), cmap='gray', norm=colors.PowerNorm(gamma=0.2))
    plt.title(f'ACS = {auto_calib_lines}')
    plt.show()

    return ACS


def compute_ssim(imageA, imageB):
    return ssim(imageA, imageB, data_range=1.0)

def compute_nmse(reference, reconstruction):
    error = reference - reconstruction
    return np.sum(error**2) / np.sum(reference**2)


def intensity_normalization(source_image, reference_image):
    """
    Adjust the intensities of the source_image to match the histogram of reference_image.

    Parameters:
    - source_image: The image whose intensities need to be adjusted.
    - reference_image: The reference image whose histogram will be matched.

    Returns:
    - matched_image: The source_image adjusted to match the histogram of reference_image.
    """
    matched_image = match_histograms(source_image, reference_image)
    return matched_image

def crop_center(img):
    '''
      n0, n1 = img.shape[:2]

      # Check if height (n0) is the larger dimension
      if n0 > n1:
          start0 = n0 // 2 - n0 // 4
          end0 = n0 // 2 + n0 // 4
          return img[start0:end0, :]

      # Otherwise, width (n1) is the larger dimension or they are equal
      start1 = n1 // 2 - n1 // 4
      end1 = n1 // 2 + n1 // 4
      return img[:, start1:end1]
    '''

    crop_factor = 3
    n0, n1 = img.shape[:2]
    start0 = n0 // 2 - n0 //crop_factor
    end0 = n0 // 2 + n0 // crop_factor - 1
    start1 = n1 // 2 - n1 // crop_factor
    end1 = n1 // 2 + n1 // crop_factor - 1
    return img[start0:end0, start1:end1]


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def calculate_psnr(mse, max_pixel_value=1.0):
    """
    Calculate PSNR from the given MSE and maximum pixel value.
    """
    if mse == 0:
        return float('inf')  # Infinite PSNR when MSE is zero
    return 10 * np.log10(max_pixel_value**2 / mse)

def normalize_image(image):
    """
    Normalize the image to range [0, 1].
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

