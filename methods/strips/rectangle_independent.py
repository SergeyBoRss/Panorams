import numpy as np

from ..base import rectangle_bounds
from .rectangle_top_bottom import StripsRectangleTopBottom


class StripsRectangleIndependent(StripsRectangleTopBottom):
    def get_settings(self, mask):
        top, bottom = rectangle_bounds(mask)
        height = mask.shape[0]
        return dict(windows=[(0, height), (0, bottom), (top, height)], rectangle_bounds=(top, bottom),
                    merge_row=(top + bottom) // 2)

    def run(self, original, mask, generate):
        height, _ = mask.shape
        top, bottom = rectangle_bounds(mask)
        common, remaining = original.copy(), mask.copy()
        frame = self.prepare_rectangle(original, mask, generate)
        if frame is not None:
            common, remaining = frame["image"].copy(), frame["remaining"].copy()
            yield frame
        current = common.copy()
        common_mask = remaining.copy()
        split = (top + bottom) // 2
        for stage, crop_top, crop_bottom in (("top", 0, bottom), ("bottom", top, height)):
            crop_height = crop_bottom - crop_top
            padding = -crop_height % 8
            pad_top, pad_bottom = (0, padding) if stage == "top" else (padding, 0)
            side = np.pad(common[crop_top:crop_bottom], ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="edge")
            side_mask = np.pad(common_mask[crop_top:crop_bottom], ((pad_top, pad_bottom), (0, 0)),
                               constant_values=False)
            for frame in super().run(side, side_mask, generate):
                side_result = common.copy()
                side_result[crop_top:crop_bottom] = frame["image"][pad_top:pad_top + crop_height]
                if stage == "top":
                    current[:split] = side_result[:split]
                else:
                    current[split:] = side_result[split:]
                accepted = frame["band"][pad_top:pad_top + crop_height]
                remaining[crop_top:crop_bottom] &= ~accepted
                yield dict(frame, stage=stage, image=current.copy(),
                           remaining=remaining.copy(), crop_top=crop_top, crop_bottom=crop_bottom,
                           pad_top=pad_top, pad_bottom=pad_bottom, merge_row=split,
                           rectangle_top=top, rectangle_bottom=bottom)
