from setuptools import setup, find_packages

setup(
    name="WDB Glyph Data",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wdb-data-rasterize = glyphdata.rasterize_font:rasterize",
            "wdb-data-dump = glyphdata.dump_rasterizations:font_dump",
            "wdb-data-augment = glyphdata.augment_rasterizations:font_augment",
            "wdb-data-stylize = glyphdata.stylize_rasterizations:run_stylization_process"
        ],
    },
)
