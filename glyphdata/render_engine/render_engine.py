## std
from typing import List, Optional
import numpy as np

## viz
from fontTools.ttLib import TTFont, TTLibError
from fontTools.unicode import Unicode
from fontTools.pens.freetypePen import FreeTypePen
from fontTools.misc.transform import Offset
from freetype.ft_errors import FT_Exception


def rasterize_glyphs(font: TTFont, alphabet: Optional[List[str]] = None) -> List[dict]:
    """
    Take a font and rasterize all contained glyphs.

    :param font: Font from which the implemented glyphs are rasterized.
    :return: Dict with images for each scale
    """
    glyphs = []

    # Retrieve available glyphs from font
    glyphset = font.getGlyphSet()

    # Rasterize glyph.py and save
    for k, v in font.getReverseGlyphMap().items():
        if alphabet is not None and k not in alphabet:
            continue
        try:
            pen = FreeTypePen(glyphset)
            glyph = glyphset[k]
            glyph.draw(pen)
            width, ascender, descender = glyph.width, font['OS/2'].usWinAscent, -font['OS/2'].usWinDescent
            height = ascender - descender
            img = pen.array(width=width, height=height, transform=Offset(0, -descender), contain=True)

            # Skip glyphs that only contain background (e.g. different types of spaces)
            if len(np.unique(img)) in [0, 1]:
                continue

            arr = np.asarray(pen.image(width=width, height=height, transform=Offset(0, -descender), contain=True))[
                ..., -1]
            glyphs.append({"char": k, "unicode": v, "img": arr})
        except SystemError as se:
            print(se)
        except FT_Exception as fte:
            print(fte)
    return glyphs
