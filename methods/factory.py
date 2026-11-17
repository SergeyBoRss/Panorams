from .growing import Growing
from .strips.strips import Strips
from .strips.top_bottom import StripsTopBottom
from .strips.rectangle import StripsRectangle
from .strips.rectangle_top_bottom import StripsRectangleTopBottom
from .strips.rectangle_independent import StripsRectangleIndependent
from .top_bottom import TopBottom
from .whole import Whole


METHODS = {
    "whole": Whole,
    "strips": Strips,
    "growing": Growing,
    "top-bottom": TopBottom,
    "strips-top-bottom": StripsTopBottom,
    "strips-rectangle-top-bottom": StripsRectangleTopBottom,
    "strips-rectangle-independent": StripsRectangleIndependent,
    "strips-rectangle": StripsRectangle,
}


def create_method(mode, blur_kernel, output_dir, strip_width=None):
    method_class = METHODS[mode]
    if method_class.uses_strip_width:
        return method_class(blur_kernel, output_dir, strip_width)
    return method_class(blur_kernel, output_dir)
