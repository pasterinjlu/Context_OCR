"""Original-resolution Gaussian blur with explicit protection of red outline pixels."""
from PIL import Image, ImageFilter
import numpy as np
import cv2


def blur_target(image, region, sigma):
    region=tuple(region)
    patch=image.crop(region)
    pixels=np.asarray(patch)
    red=(pixels[:,:,0]>100)&(pixels[:,:,0].astype(float)>1.35*pixels[:,:,1])&(pixels[:,:,0].astype(float)>1.35*pixels[:,:,2])
    protected=cv2.dilate(red.astype('uint8'),np.ones((3,3),dtype='uint8'))
    mask=Image.fromarray((255*(1-protected)).astype('uint8'))
    output=image.copy()
    output.paste(patch.filter(ImageFilter.GaussianBlur(sigma)),region[:2],mask)
    return output, int(red.sum())


def inside_bbox(
    bbox_xywh: list[int], image_size: tuple[int, int], inset_ratio: float, min_inset: int
) -> tuple[int, int, int, int]:
    if len(bbox_xywh) != 4:
        raise ValueError(f"Invalid bbox: {bbox_xywh}")
    x, y, width, height = (int(value) for value in bbox_xywh)
    inset = max(min_inset, int(round(height * inset_ratio)))
    left, top = x + inset, y + inset
    right, bottom = x + width - inset, y + height - inset
    if left < 0 or top < 0 or right > image_size[0] or bottom > image_size[1]:
        raise ValueError(f"BBox outside image: {bbox_xywh} vs {image_size}")
    if right - left < 4 or bottom - top < 4:
        raise ValueError(f"Blur interior is too small: {bbox_xywh}")
    return left, top, right, bottom
