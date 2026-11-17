import cv2
import numpy as np

from ..base import Method


def next_band(remaining, strip_width):
    if strip_width < 1:
        raise ValueError("strip_width must be positive")
    if remaining.all():
        raise ValueError("The mask needs some known pixels")
    distance = cv2.distanceTransform(remaining.astype(np.uint8), cv2.DIST_L2, 5)
    return remaining & (distance <= strip_width)


def blend_band(current, generated, remaining, band, blur_kernel):
    mask_np = band.astype(np.float64)[:, :, None]
    mask_blurred = cv2.GaussianBlur(mask_np * 255, (blur_kernel, blur_kernel), 0) / 255
    mask_blurred = mask_blurred[:, :, None]
    alpha = 1 - (1 - mask_np) * (1 - mask_blurred)
    alpha[remaining & ~band] = 0
    return (current * (1 - alpha) + generated * alpha).astype(np.uint8)


class Strips(Method):
    uses_strip_width = True

    def __init__(self, blur_kernel, output_dir, strip_width):
        super().__init__(blur_kernel, output_dir)
        self.strip_width = strip_width

    def run(self, original, mask, generate):
        current, remaining = original.copy(), mask.copy()
        while remaining.any():
            missing = remaining.copy()
            band = next_band(missing, self.strip_width)
            generated = np.asarray(generate(current.copy(), missing))
            current = blend_band(current, generated, missing, band, self.blur_kernel)
            remaining &= ~band
            yield dict(stage="whole", bottom=mask.shape[0], image=current.copy(), generated=generated,
                       mask=missing, band=band, remaining=remaining.copy(),
                       stage_complete=not remaining.any())

    def save_frame(self, frame, index, source_size, history):
        super().save_frame(frame, index, source_size, history)
        history[-1].update(stage=frame["stage"], accepted_pixels=int(frame["band"].sum()),
                           stage_complete=bool(frame["stage_complete"]))
        history[-1].update({key: frame[key] for key in
                           ("crop_top", "crop_bottom", "pad_top", "pad_bottom", "merge_row") if key in frame})
