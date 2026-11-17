import numpy as np
from PIL import Image

from .base import Method, top_bottom_windows


class TopBottom(Method):
    def get_settings(self, mask):
        return dict(windows=top_bottom_windows(mask))

    def run(self, original, mask, generate):
        current = original.copy()
        for top, bottom in top_bottom_windows(mask):
            crop = original[top:bottom].copy()
            crop_mask = mask[top:bottom].copy()
            padding = -(bottom - top) % 8
            model_input = np.pad(crop, ((0, padding), (0, 0), (0, 0)), mode="edge")
            model_mask = np.pad(crop_mask, ((0, padding), (0, 0)), mode="edge")
            generated = None
            if crop_mask.any():
                generated = np.asarray(generate(model_input, model_mask))
                crop[crop_mask] = generated[:bottom - top][crop_mask]
            current[top:bottom][crop_mask] = crop[crop_mask]
            yield dict(top=top, bottom=bottom, input=model_input, mask=model_mask,
                       pad_bottom=padding, generated=generated, image=current.copy())

    def save_frame(self, frame, index, source_size, history):
        width, height = source_size
        Image.fromarray(frame["image"][:height, :width]).save(self.output_dir / f"step_{index:03d}.png")
        stage = "top" if index == 0 else "bottom"
        if frame["generated"] is not None:
            history[-1].update(stage=stage, crop_top=frame["top"], crop_bottom=frame["bottom"],
                               pad_bottom=frame["pad_bottom"], step_index=index)
