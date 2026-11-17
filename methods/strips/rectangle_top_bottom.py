import numpy as np

from ..base import rectangle_bounds
from .rectangle import StripsRectangle
from .strips import blend_band


class StripsRectangleTopBottom(StripsRectangle):
    def get_settings(self, mask):
        top, bottom = rectangle_bounds(mask)
        height = mask.shape[0]
        return dict(windows=[(0, height), (0, (bottom + 7) // 8 * 8), (0, height)],
                    rectangle_bounds=(top, bottom))

    def run(self, original, mask, generate):
        height, _ = mask.shape
        top, bottom = rectangle_bounds(mask)
        current, remaining = original.copy(), mask.copy()
        frame = self.prepare_rectangle(original, mask, generate)
        if frame is not None:
            current, remaining = frame["image"].copy(), frame["remaining"].copy()
            yield frame

        def accept(stage, band, crop_bottom, stage_complete):
            missing = remaining[:crop_bottom].copy()
            padding = -crop_bottom % 8
            model_image = np.pad(current[:crop_bottom], ((0, padding), (0, 0), (0, 0)), mode="edge")
            model_mask = np.pad(missing, ((0, padding), (0, 0)), constant_values=False)
            generated = np.asarray(generate(model_image, model_mask))
            if generated.shape != model_image.shape or generated.dtype != np.uint8:
                raise ValueError("Generator must return uint8 RGB at the current stage size")
            current[:crop_bottom] = blend_band(current[:crop_bottom], generated[:crop_bottom],
                                              missing, band[:crop_bottom], self.blur_kernel)
            remaining[band] = False
            return dict(stage=stage, bottom=crop_bottom + padding, image=current.copy(), generated=generated,
                        mask=model_mask, band=np.pad(band[:crop_bottom], ((0, padding), (0, 0))),
                        remaining=remaining.copy(), stage_complete=stage_complete,
                        rectangle_top=top, rectangle_bottom=bottom)

        frontier = top
        while frontier > 0:
            next_frontier = max(0, frontier - self.strip_width)
            band = np.zeros_like(mask)
            band[next_frontier:frontier] = True
            yield accept("top", band, bottom, next_frontier == 0)
            frontier = next_frontier

        frontier = bottom
        while frontier < height:
            next_frontier = min(height, frontier + self.strip_width)
            band = np.zeros_like(mask)
            band[frontier:next_frontier] = True
            yield accept("bottom", band, height, next_frontier == height)
            frontier = next_frontier
