import numpy as np

from ..base import rectangle_bounds
from .strips import Strips, blend_band


class StripsRectangle(Strips):
    def get_settings(self, mask):
        return dict(windows=[(0, mask.shape[0]), (0, mask.shape[0])],
                    rectangle_bounds=rectangle_bounds(mask))

    def prepare_rectangle(self, original, mask, generate):
        top, bottom = rectangle_bounds(mask)
        band = mask.copy()
        band[:top] = False
        band[bottom:] = False
        if not band.any():
            return None
        generated = np.asarray(generate(original.copy(), mask))
        if generated.shape != original.shape or generated.dtype != np.uint8:
            raise ValueError("Generator must return uint8 RGB at the current stage size")
        current = blend_band(original, generated, mask, band, self.blur_kernel)
        remaining = mask & ~band
        return dict(stage="rectangle", bottom=mask.shape[0], image=current, generated=generated,
                    mask=mask.copy(), band=band, remaining=remaining, stage_complete=True,
                    rectangle_top=top, rectangle_bottom=bottom)

    def run(self, original, mask, generate):
        current, remaining = original.copy(), mask.copy()
        frame = self.prepare_rectangle(original, mask, generate)
        if frame is not None:
            current, remaining = frame["image"].copy(), frame["remaining"].copy()
            yield frame
        for frame in super().run(current, remaining, generate):
            yield dict(frame, stage="strips")
