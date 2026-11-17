import argparse
from functools import partial
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from .model import load_model, generate_image
from .methods.factory import METHODS, create_method


logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    common = parser.add_argument_group("Common arguments")
    common.add_argument("--image", type=Path, default=Path("inputs/pano_01/image.png"))
    common.add_argument("--mask", type=Path, default=Path("inputs/pano_01/mask.png"))
    common.add_argument("--output-dir", type=Path, default=Path("outputs/pano_01_original_size"))
    common.add_argument("--mode", choices=tuple(METHODS), required=True)
    common.add_argument("--steps", type=int, default=50)
    common.add_argument("--seed", type=int, default=1234)
    common.add_argument("--device", default="cuda")
    common.add_argument("--prompt", default="")
    common.add_argument("--negative-prompt", default="")
    common.add_argument("--guidance", type=float, default=1.0)
    common.add_argument("--control", type=float, default=1.0)
    common.add_argument("--blur-kernel", type=int, default=21)
    common.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--no-model-cpu-offload", dest="model_cpu_offload", action="store_false")
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--base-model", type=Path, default=Path("data/ckpt/realisticVisionV60B1_v51VAE"))
    common.add_argument("--brushnet", type=Path, default=Path("data/ckpt/random_mask_brushnet_ckpt"))

    strips = parser.add_argument_group("Strip and growing modes")
    strips.add_argument("--strip-width", default=64, metavar="PIXELS")

    args = parser.parse_args(argv)

    method_class = METHODS[args.mode]
    if not method_class.uses_strip_width:
        del args.strip_width
    else:
        try:
            args.strip_width = int(args.strip_width)
        except ValueError:
            parser.error("--strip-width must be an integer")
        if args.strip_width < 1:
            parser.error("--strip-width must be positive")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def pad_inputs(original, mask):
    if original.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions must match")
    height, width = mask.shape
    padding = ((0, -height % 8), (0, -width % 8))
    return (np.pad(original, padding + ((0, 0),), mode="edge"),
            np.pad(mask, padding, mode="edge"))


def generate_frame(image, missing, *, pipe, args, history):
    seed = args.seed
    call = len(history) + 1
    record = dict(call=call, width=image.shape[1], height=image.shape[0],
                  masked_pixels=int(missing.sum()), seed=seed, status="running")
    history.append(record)
    write_json(args.output_dir / "frames.json", history)
    logger.info("Generation %s: %sx%s, seed=%s", call, image.shape[1], image.shape[0], seed)
    generated = generate_image(
        pipe, image, missing, seed=seed,
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        steps=args.steps, guidance=args.guidance, control=args.control,
    )
    record["status"] = "complete"
    write_json(args.output_dir / "frames.json", history)
    return generated


def run_mode(args):
    with Image.open(args.image) as image:
        original = np.array(image.convert("RGB"))
    with Image.open(args.mask) as image:
        mask = np.array(image.convert("L")) == 255
    height, width = original.shape[:2]
    original, mask = pad_inputs(original, mask)

    method = create_method(
        args.mode, blur_kernel=args.blur_kernel, output_dir=args.output_dir,
        strip_width=args.strip_width if METHODS[args.mode].uses_strip_width else None,
    )
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if value is not None
    }
    settings.update(source_size=[width, height], padded_size=[original.shape[1], original.shape[0]])
    settings.update(method.get_settings(mask))
    logger.info("%s", json.dumps(settings, ensure_ascii=False))
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "settings.json", settings)
    started = time.monotonic()
    history = []
    try:
        pipe = None
        if mask.any():
            pipe = load_model(
                base_model=args.base_model, brushnet=args.brushnet, device=args.device,
                model_cpu_offload=args.model_cpu_offload, vae_tiling=args.vae_tiling,
            )

        generate = partial(generate_frame, pipe=pipe, args=args, history=history)

        current = original.copy()
        for index, frame in enumerate(method.run(original, mask, generate)):
            current = frame["image"]
            method.save_frame(frame, index, (width, height), history)
            if history:
                write_json(args.output_dir / "frames.json", history)
        Image.fromarray(current[:height, :width]).save(args.output_dir / "final.png")
        status = dict(status="complete", calls=len(history), seconds=time.monotonic() - started)
    except Exception as error:
        status = dict(status="failed", error_type=type(error).__name__, error=str(error),
                      calls=len(history), seconds=time.monotonic() - started)
        write_json(args.output_dir / "status.json", status)
        logger.exception("Generation failed for mode %s", args.mode)
        return 1
    write_json(args.output_dir / "status.json", status)
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    return run_mode(args)


if __name__ == "__main__":
    sys.exit(main())
