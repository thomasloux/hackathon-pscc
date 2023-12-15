import numpy as np
import cv2

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if np.array_equal(img, np.zeros(img.shape)):
      return 0

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: str, shape,label=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1] * shape[2], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction

def find_largest_containing_circle(segmentation, pixdim):
    largest_circle = None
    largest_slice = -1
    max_radius = -1

    segmentation8 = segmentation.astype(np.float32).astype('uint8')
    for i in range(segmentation8.shape[-1]):
        # Find the contours in the segmentation
        contours, _ = cv2.findContours(image = segmentation8[:,:,i], mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Fit the smallest circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius > max_radius:
                max_radius = radius
                largest_circle = ((int(x), int(y)), int(radius))
                largest_slice = i
    recist = max_radius * 2 * pixdim[0]
#     print(max_radius)
    predicted_volume = np.round(np.sum(segmentation.flatten())*pixdim[0]*pixdim[1]*pixdim[2]*0.001,2)
    return recist, predicted_volume, largest_circle, largest_slice
