import click
import tqdm
import time
from pathlib import Path
from fontTools.ttLib import TTFont, TTLibError

# custom
from glyphdata.render_engine.render_engine import rasterize_glyphs
from glyphdata.utils.io_utils import compressed_pickle

@click.command()
@click.option(
    "--font-dir", "-f",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory that contains the font files (.otf, .ttf) that should be rasterized."
)
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory where rasterized font files will be saved."
)

def rasterize(font_dir, output_dir):
    suffix = ["otf", "ttf"]
    files = []
    for s in suffix:
        for p in Path(font_dir).rglob(f"*.{s}"):
            files.append(p)

    click.echo(f"Starting to rasterize {len(files)} font files in `{font_dir}`.")

    for fn in (pbar := tqdm.tqdm(files)):
        try:
            font = TTFont(fn.as_posix())
            font_family, font_name = font['name'].getDebugName(1), font['name'].getDebugName(4)
            tdir = Path(output_dir).joinpath(fn.relative_to(font_dir).parent)
            tpath = tdir.joinpath(font_name + '.pbz2')
            if tpath.exists():
                continue

            # 1) Rasterize glyphs
            pbar.set_description(f"Rasterizing glyphs for font: '{font_name}'")
            glyph_data = rasterize_glyphs(font)

            # 2) Add meta data
            out = {"font_name": font_name,
                   "font_family": font_family,
                   "timestamp": time.time(),
                   "glyphs": glyph_data}

            # 3) Save compressed pickle file
            pbar.set_description(f"Saving glyphs for font: '{font_name}'")
            tdir.mkdir(parents=True, exist_ok=True)
            compressed_pickle(out, tpath.as_posix())

        except TTLibError as ttle:
            pbar.set_description(f"Skipping due to TTLibError {ttle}")
            continue

    click.echo(f"Finished font rasterization job. Files successfully saved at {output_dir}.")




if __name__ == '__main__':
    rasterize()
