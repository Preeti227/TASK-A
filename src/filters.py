import cv2
import numpy as np

def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    return np.clip(sobel / sobel.max(), 0, 1).astype(np.float32)

def apply_laplacian(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return np.clip(lap / lap.max(), 0, 1).astype(np.float32)

def apply_salt_pepper_noise(image, amount=0.02, s_vs_p=0.5):
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * amount * s_vs_p)
    num_pepper = int(total_pixels * amount * (1.0 - s_vs_p))
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[tuple(coords_salt)] = 1.0
    noisy[tuple(coords_pepper)] = 0.0
    return noisy

def apply_all_filters(img, img_size=(224, 224)):
    img = cv2.resize(img, img_size)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img
    img_norm = img_gray.astype(np.float32) / 255.0
    filters = [
        apply_gaussian_blur(img_gray),
        apply_sobel(img_gray),
        apply_laplacian(img_gray),
        apply_salt_pepper_noise(img_norm)
    ]
    return np.stack(filters, axis=-1)
