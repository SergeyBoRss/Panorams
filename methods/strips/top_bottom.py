import numpy as np

from ..base import top_bottom_windows
from .strips import Strips


class StripsTopBottom(Strips):
    def get_settings(self, mask):
        windows = top_bottom_windows(mask) if mask.any() else []
        settings = dict(windows=windows)
        if windows:
            settings["merge_row"] = (windows[1][0] + windows[0][1]) // 2
        return settings

    def run(self, original, mask, generate):
        if not mask.any():
            return
        windows = top_bottom_windows(mask)
        split = (windows[1][0] + windows[0][1]) // 2
        current, remaining = original.copy(), mask.copy()
        for stage, (top, bottom) in zip(("top", "bottom"), windows):
            crop_height = bottom - top
            padding = -crop_height % 8
            pad_top, pad_bottom = (0, padding) if stage == "top" else (padding, 0)
            side = np.pad(original[top:bottom], ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="edge")
            side_mask = np.pad(mask[top:bottom], ((pad_top, pad_bottom), (0, 0)), constant_values=False)
            for frame in super().run(side, side_mask, generate):
                side_result = original.copy()
                side_result[top:bottom] = frame["image"][pad_top:pad_top + crop_height]
                if stage == "top":
                    current[:split] = side_result[:split]
                else:
                    current[split:] = side_result[split:]
                accepted = frame["band"][pad_top:pad_top + crop_height]
                remaining[top:bottom] &= ~accepted
                yield dict(frame, stage=stage, image=current.copy(),
                           remaining=remaining.copy(), crop_top=top, crop_bottom=bottom,
                           pad_top=pad_top, pad_bottom=pad_bottom, merge_row=split)
