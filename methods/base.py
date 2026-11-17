from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image


def rectangle_bounds(mask):
    rows = np.flatnonzero((~mask).any(axis=1))
    if not rows.size:
        raise ValueError("The mask needs some known pixels")
    return int(rows[0]), int(rows[-1]) + 1


def top_bottom_windows(mask):
    height = mask.shape[0]
    if not mask.any():
        return [(0, height)]
    known_rows = np.flatnonzero(~mask.any(axis=1))
    if not known_rows.size:
        raise ValueError("top-bottom modes need a fully known horizontal row")
    bands = np.split(known_rows, np.flatnonzero(np.diff(known_rows) > 1) + 1)
    context = max(bands, key=len)
    return [(0, int(context[-1]) + 1), (int(context[0]), height)]


def blurred_blend(original, generated, mask, kernel_size):
    mask_np = mask.astype(np.float64)[:, :, None]
    mask_blurred = cv2.GaussianBlur(mask_np * 255, (kernel_size, kernel_size), 0) / 255
    mask_blurred = mask_blurred[:, :, None]
    alpha = 1 - (1 - mask_np) * (1 - mask_blurred)
    return (original * (1 - alpha) + generated * alpha).astype(np.uint8)


class Method(ABC):
    uses_strip_width = False

    def __init__(self, blur_kernel, output_dir):
        self.blur_kernel = blur_kernel
        self.output_dir = output_dir

    def get_settings(self, mask):
        return {"windows": [(0, mask.shape[0])]}

    @abstractmethod
    def run(self, original, mask, generate):
        raise NotImplementedError

    def save_frame(self, frame, index, source_size, history):
        width, height = source_size
        Image.fromarray(frame["image"][:height, :width]).save(
            self.output_dir / f"step_{len(history):03d}.png")
