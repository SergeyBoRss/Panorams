from .base import Method, blurred_blend


class Whole(Method):
    def run(self, original, mask, generate):
        if not mask.any():
            return
        generated = generate(original.copy(), mask)
        yield {"image": blurred_blend(original, generated, mask, self.blur_kernel)}
