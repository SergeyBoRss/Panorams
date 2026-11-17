import numpy as np
from PIL import Image

from .base import Method, blurred_blend, rectangle_bounds


class Growing(Method):
    uses_strip_width = True

    def __init__(self, blur_kernel, output_dir, strip_width):
        super().__init__(blur_kernel, output_dir)
        self.strip_width = strip_width

    def windows(self, mask):
        top, bottom = rectangle_bounds(mask)
        windows = [(top, bottom)]
        while top > 0 or bottom < mask.shape[0]:
            top = max(0, top - self.strip_width)
            bottom = min(mask.shape[0], bottom + self.strip_width)
            windows.append((top, bottom))
        return windows

    def get_settings(self, mask):
        return {"windows": self.windows(mask)}

    def run(self, original, mask, generate):
        current, remaining = original.copy(), mask.copy()
        for top, bottom in self.windows(mask):
            crop = current[top:bottom].copy()
            crop_mask = remaining[top:bottom].copy()
            padding = -(bottom - top) % 8
            model_input = np.pad(crop, ((0, padding), (0, 0), (0, 0)), mode="edge")
            model_mask = np.pad(crop_mask, ((0, padding), (0, 0)), mode="edge")
            generated = None
            if crop_mask.any():
                generated = np.asarray(generate(model_input, model_mask))
                crop = blurred_blend(crop, generated[:bottom - top], crop_mask, self.blur_kernel)
            current[top:bottom] = crop
            remaining[top:bottom] = False
            yield dict(top=top, bottom=bottom, input=model_input, mask=model_mask,
                       pad_bottom=padding, generated=generated, image=crop)

    def save_frame(self, frame, index, source_size, history):
        width, _ = source_size
        Image.fromarray(frame["image"][:, :width]).save(self.output_dir / f"step_{index:03d}.png")
