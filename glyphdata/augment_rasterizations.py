from pathlib import Path

import click
import tqdm
from glyphdata.augmentation_engine.augmentation_engine import augment_glyph
from glyphdata.glyph.glyph import Glyph
from glyphdata.utils.io_utils import decompress_pickle, validate_positive_int


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
    '--num-augmentations', "-n",
    required=True,
    type=int,
    callback=validate_positive_int,
    help='The number of augmented version that shall be generated per rasterized character.'
)
@click.option(
    '--image-size', "-s",
    required=False,
    default=128,
    type=int,
    callback=validate_positive_int,
    help='Square image size as a single integer (e.g., 128)'
)
def font_augment(raster_dir, output_dir, num_augmentations, image_size):
    odir_sized = Path(output_dir).joinpath("_sized")
    odir_residual = Path(output_dir).joinpath("_residual")
    odir_composite = Path(output_dir).joinpath("composite")
    odir_sized.mkdir(exist_ok=True, parents=True)
    odir_residual.mkdir(exist_ok=True, parents=True)
    odir_composite.mkdir(exist_ok=True, parents=True)

    for fn in tqdm.tqdm(list(Path(raster_dir).rglob("*.pbz2"))):
        ds: dict = decompress_pickle(fn)
        font_name = ds['font_name']
        font_family = ds['font_family']
        data = ds['glyphs']
        residuals = [g_data['img'] for g_data in data]

        for g_data in tqdm.tqdm(data, desc=f"Processing: {fn.name}"):
            char, unicode, img = g_data['char'], g_data['unicode'], g_data['img']
            g = Glyph(char, unicode, font_name, font_family, img)
            g.compute_sized_img(size=image_size, backend="skimage")
            augmentation_results = augment_glyph(char,
                                                 unicode,
                                                 g.sized_img_white_on_black,
                                                 residuals,
                                                 num_augmentations,
                                                 image_size)

            for (char_img, res_img, comp_img, tname) in zip(*augmentation_results):
                oname = f"{font_name}_{tname}"

                Glyph.dump_arr(char_img, odir_sized, oname, "png", True)
                Glyph.dump_arr(res_img, odir_residual, oname, "png", True)
                Glyph.dump_arr(comp_img, odir_composite, oname, "png", True)

if __name__ == '__main__':
    font_augment()
