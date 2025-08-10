import click
import tqdm
import time
from pathlib import Path
from fontTools.ttLib import TTFont, TTLibError

# custom
from glyphdata.render_engine.render_engine import rasterize_glyphs
from glyphdata.utils.io_utils import decompress_pickle
from glyphdata.glyph.glyph import Glyph


def validate_image_size(ctx, param, value):
    """Validate that the input is a valid integer."""
    if value <= 0:
        raise click.BadParameter("The image size must be a positive integer.")
    return value


@click.command()
@click.option(
    "--raster-dir", "-r",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory that contains the rasterized files (.pbz2) for which an image dump should be performed."
)
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory where image dump will be saved."
)
@click.option(
    '--image-size', "-s",
    required=False,
    default=128, 
    type=int, 
    callback=validate_image_size,
    help='Square image size as a single integer (e.g., 128)'
)
@click.option(
    "--output-format", "-f",
    required=False,
    type=click.Choice(["image_only", "skeleton_only", "all"], case_sensitive=False),
    default = "all",
    help="Output format: one of 'image_only', 'skeleton_only', or 'all'."
)
@click.option(
    "--color-mode", "-c",
    required=False,
    type=click.Choice(["black_on_white", "white_on_black"], case_sensitive=False),
    default = "black_on_white",
    help="Color made: one of 'black_on_white' or 'skelewhite_on_blackton_only'."
)
def font_dump(raster_dir, output_dir, image_size, output_format, color_mode):
    odir_img = Path(output_dir).joinpath("img")
    odir_skel = Path(output_dir).joinpath("skel")
    odir_img.mkdir(exist_ok=True, parents=True)
    odir_skel.mkdir(exist_ok=True, parents=True)

    for fn in tqdm.tqdm(list(Path(raster_dir).rglob("*.pbz2"))):
        ds: dict = decompress_pickle(fn)
        font_name = ds['font_name']
        font_family = ds['font_family']
        data = ds['glyphs']

        for g_data in data:
            char, unicode, img = g_data['char'], g_data['unicode'], g_data['img']

            g = Glyph(char, unicode, font_name, font_family, img)
            g.compute_sized_img(size=image_size,
                                pad_val=0,
                                backend="skimage")
            g.compute_skelgraph(skeleton_method="zhang", 
                                invert=False, 
                                num_sample_points=16, 
                                degree=3)
            if g.graph_valid:
                bow = color_mode == "black_on_white"
                if output_format in ("image_only", "all"):
                    g.dump_image(odir_img, black_on_white=bow)
                if output_format in ("skeleton_only", "all"):
                    g.dump_skeleton(odir_skel, black_on_white=bow)


if __name__ == '__main__':
    font_dump()
