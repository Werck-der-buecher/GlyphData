## std
import dataclasses

import cv2
import bz2
import pickle as cPickle
import numpy as np
from typing import Union, Sequence, List, Tuple, Optional, Literal
from dataclasses import dataclass, field

## viz
import matplotlib.axis
from scipy.interpolate import splprep, splev
from scipy.sparse import triu
import matplotlib.pyplot as plt

## misc
import albumentations as A
import skan
import pathlib
from skimage.transform import rescale
from skimage.util import img_as_ubyte

## custom
from glyphdata.algorithms import get_skel_method
from glyphdata.utils.data_utils import minmax
from glyphdata.utils.curve_utils import BezierCurve


@dataclasses.dataclass
class StylizedImage(object):
    style_img: np.ndarray

    # Meta information
    style_method: str

    # Optional
    style_reference_name: Optional[str] = None
    style_reference_size: Optional[Tuple[int]] = None
    style_reference_img: Optional[np.ndarray] = None


class Glyph(object):
    # CONST
    char: str
    unicode: int
    font: str
    font_family: str
    augmentation_id: str
    _orig_img: np.ndarray

    # DYNAMIC
    ## Img
    _sized_img: Optional[np.ndarray] = None
    _size: Optional[Sequence[int]] = None

    ## Style transfer image(s)
    _stylizations: Optional[Union[List[dict], List[StylizedImage]]] = None

    ## Skeleton
    _skeleton: Optional[np.ndarray] = None
    _skeleton_method: Optional[str] = None

    ## Skeleton graph
    _nodes: Optional[np.ndarray] = None
    _edges: Optional[np.ndarray] = None
    _degrees: Optional[np.ndarray] = None

    ## Junction graph skel_data
    _jnodes: Optional[np.ndarray] = None
    _jedges: Optional[np.ndarray] = None
    _jcurves: Optional[np.ndarray] = None
    _jcurves_tck: Optional[list] = None
    _num_sample_points: Optional[int] = None

    ## Validation flags
    _fimg = False
    _fstyle = False
    _fskel = False
    _fgraph = False

    def __init__(self,
                 char: str,
                 unicode: int,
                 font: str,
                 font_family: str,
                 img: Optional[np.ndarray] = None,
                 sized_img: Optional[np.ndarray] = None,
                 augmentation_id: Optional[str] = None,
                 stylizations: Optional[Union[List[dict], List[StylizedImage]]] = None):
        self.char = char
        self.unicode = unicode
        self.font = font
        self.font_family = font_family
        self.augmentation_id = augmentation_id
        self._orig_img = img
        self._sized_img = sized_img
        self._stylizations = stylizations

        assert self._orig_img is not None or self._sized_img is not None

    @property
    def orig_img(self):
        return self._orig_img

    @property
    def stylizations(self):
        return self._stylizations

    @stylizations.setter
    def stylizations(self, stylizations):
        self._stylizations = stylizations

    @property
    def sized_img(self):
        if self._sized_img is None:
            raise AssertionError("Resized image was not yet computed. Please invoke 'self.compute_for_size'.")

        return self._sized_img

    @property
    def sized_img_white_on_black(self):
        return self.sized_img

    @property
    def sized_img_black_on_white(self):
        return 1 - self.sized_img

    @property
    def skeleton(self):
        if self._skeleton is None:
            raise AssertionError("Skeleton was not yet computed. Please invoke 'self.compute_for_size'.")
        return self._skeleton

    @property
    def skeleton_method(self):
        if self._skeleton_method is None:
            raise AssertionError("Skeleton was not yet computed. Please invoke 'self.compute_for_size'.")
        return self._skeleton

    @property
    def skeleton_nodes(self):
        if self._nodes is None:
            raise AssertionError("Skeleton graph nodes were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._nodes

    @property
    def skeleton_edges(self):
        if self._edges is None:
            raise AssertionError("Skeleton graph edges were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._edges

    @property
    def skeleton_degrees(self):
        if self._degrees is None:
            raise AssertionError(
                "Skeleton graph node degrees were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._degrees

    @property
    def junction_nodes(self):
        if self._jnodes is None:
            raise AssertionError("Junction graph nodes were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._jnodes

    @property
    def junction_edges(self):
        if self._jedges is None:
            raise AssertionError("Junction graph edges were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._jedges

    @property
    def junction_bezcurves_ctrl_points(self):
        if self._jbezier_cp is None:
            raise AssertionError("Junction graph curves were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._jbezier_cp

    @property
    def junction_bezcurves_sample_points(self):
        if self._jbezier_sp is None:
            raise AssertionError("Junction graph curves were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._jbezier_sp

    @property
    def num_sample_points(self):
        if self._num_sample_points is None:
            raise AssertionError("Junction graph curves were not yet computed. Please invoke 'self.compute_for_size'.")
        return self._num_sample_points

    @property
    def sized_img_valid(self):
        return self._fimg

    @property
    def skel_valid(self):
        return self._fskel

    @property
    def graph_valid(self):
        return self._fgraph

    def compute_sized_img(self,
                          size: Union[int, Sequence[int]],
                          pad_val: int = 0,
                          backend: Literal['skimage', 'cv2', 'albumenations'] = 'skimage'):
        # Resize image
        if backend == "skimage":
            resized = self._cmpt_sized_image_skimage(self.orig_img, size, pad_val)
        elif backend in ['cv2', 'albumenations']:
            resized = self._cmpt_sized_image_cv2(self.orig_img, size, pad_val)
        else:
            raise ValueError("Backend not supported.")
        self._sized_img = resized
        self._size = resized.shape
        self._fimg = True

    def compute_skelgraph(self,
                         skeleton_method: Literal["zhang", "lee", "medial_axis", "aof"] = "zhang",
                         invert: bool = False,
                         num_sample_points: int = 16,
                         degree: int = 3):

        # Compute skeleton
        skeleton = Glyph.cmpt_skeleton(self.sized_img, skeleton_method, invert)
        self._skeleton = skeleton
        self._skeleton_method = skeleton_method
        self._fskel = True

        # Generate graph representation (fails for n <=2)
        try:
            graph = self._cmpt_graph(skeleton, num_sample_points, degree)
            self._nodes, self._edges, self._degrees, self._jnodes, self._jedges, self._jbezier_cp, self._jbezier_sp = graph
            self._num_sample_points = num_sample_points
            self._degree = degree
            self._fgraph = True
        except ValueError as ve:
            if self.augmentation_id is not None:
                lbl = f"{self.font}_{self.char}_{self.augmentation_id}_{self.unicode}"
            else:
                lbl = f"{self.font}_{self.char}_{self.unicode}"
            print(f"{lbl}: Cannot compute graph structure due to: ", ve)
            self._fgraph = False
            self._nodes, self._edges, self._degrees, = None, None, None
            self._jnodes, self._jedges, self._jbezier_cp, self._jbezier_sp = None, None, None, None
            self._num_sample_points = None
            self._degree = None

    def _cmpt_sized_image_skimage(self, img: np.ndarray, size: int, pad_val: int = 0):
        # Scaling
        height, width = img.shape[:2]
        scale = size / float(max(width, height))
        if scale != 1.0:
            img = img_as_ubyte(rescale(img, scale))

        # Padding
        y_pad = (size - img.shape[0])
        x_pad = (size - img.shape[1])
        img = np.pad(img, ((y_pad // 2, y_pad // 2 + y_pad % 2),
                           (x_pad // 2, x_pad // 2 + x_pad % 2)),
                     mode='constant', constant_values=(pad_val, pad_val))

        # Normalization
        img = minmax(img)
        return img

    def _cmpt_sized_image_cv2(self, img: np.ndarray, size: Union[int, Sequence[int]], pad_val: int = 0):
        H = size[0] if isinstance(size, Sequence) else size
        W = size[1] if isinstance(size, Sequence) else size

        if H > self.orig_img.shape[0] or W > self.orig_img.shape[1]:
            raise UserWarning(f"Specified size {size} is larger than the original image!")

        atransforms = A.Compose([
            A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=H, min_width=W, border_mode=cv2.BORDER_CONSTANT, value=pad_val),
            A.Lambda(name='min_max', image=minmax)
        ])

        transformed = atransforms(image=img)
        return transformed['image']

    @staticmethod
    def cmpt_skeleton(img: np.ndarray, skel_method: Literal["zhang", "lee", "medial_axis", "aof"] = "zhang",
                      invert=False):
        sm = get_skel_method(skel_method)(invert=invert)
        return sm.compute(img)

    def _cmpt_graph(self, skeleton: np.ndarray, num_sample_points: int = 16, degree=3):
        # Skeleton analysis
        skan_skeleton = skan.Skeleton(skeleton)
        branch_data = skan.summarize(skan_skeleton)
        nodes = skan_skeleton.coordinates
        degrees = skan_skeleton.degrees
        m = triu(skan_skeleton.graph)
        edges = np.c_[m.row, m.col]

        # Junction graph
        # jnodes_alt = nodes[degrees != 2]
        jnodes = np.unique(np.r_[branch_data[["coord-src-0", "coord-src-1"]].to_numpy(),
        branch_data[["coord-dst-0", "coord-dst-1"]].to_numpy()],
                           axis=0)

        # map full-graph indices to junction graph indices
        jedges = []
        for sn, en in branch_data[["node-id-src", "node-id-dst"]].to_numpy():
            sn_idx = np.asarray(np.all(jnodes == nodes[sn], axis=1)).nonzero()[0].item()
            en_idx = np.asarray(np.all(jnodes == nodes[en], axis=1)).nonzero()[0].item()
            jedges.append([sn_idx, en_idx])
        jedges = np.array(jedges)

        bezier_control_points = []
        bezier_sample_points = []
        for lidx in branch_data.index:
            path_coords = skan_skeleton.path_coordinates(lidx)
            coords_x = path_coords[:, 1]
            coords_y = path_coords[:, 0]

            # if len(coords_x) < degree:
            #    continue

            bc = BezierCurve(order=degree, fix_start_end=True)
            bc.get_control_points(coords_x, coords_y, interpolate=True)  # fit curve and calculate control points
            cp = bc.control_points
            sp = bc.get_sample_point(num_sample_points)
            bezier_control_points.append(cp)
            bezier_sample_points.append(sp)

            # Cubic spline interpolation, boils down to quadratic or linear if the number of skel_data points are not
            # sufficient
            # k = min(len(path_coords) - 1, 3)
            # tck, u = splprep([coords_x, coords_y], s=0, k=k)
            # Evaluate spline at a fixed number of equidistant time-steps
            # cp = splev(np.linspace(0, 1, num_sample_points), tck)
            # curve_points.append(np.array(cp).T)
            # curve_tck.append(tck)
        bezier_control_points = np.stack(bezier_control_points)
        bezier_sample_points = np.stack(bezier_sample_points)

        return nodes, edges, degrees, jnodes, jedges, bezier_control_points, bezier_sample_points

    def visualize(self, ax: Optional[matplotlib.axis.Axis] = None, set_title=False):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(1, 1)

        if self._fimg:
            ax.imshow(self.sized_img, cmap="Greys")
            if self._fgraph:
                ax.scatter(self.skeleton_nodes[:, 1], self.skeleton_nodes[:, 0], c='white', s=10, alpha=.9)
                ax.scatter(self.junction_nodes[:, 1], self.junction_nodes[:, 0], c='r')

                for e in self.junction_edges:
                    ax.plot(self.junction_nodes[e][:, 1], self.junction_nodes[e][:, 0], c='b')

                for cp in self.junction_bezcurves_ctrl_points:
                    ax.plot(cp[:, 0], cp[:, 1], c='orange', alpha=0.4)

                for cp in self.junction_bezcurves_sample_points:
                    ax.scatter(cp[:, 0], cp[:, 1], c='orange', s=10)
        else:
            ax.imshow(self.orig_img)

        if set_title:
            ax.set_title(f"Font: {self.font}\nCharacter: {self.char}")

        if standalone:
            plt.show()

    def dump_image(self, tdir: Union[str, pathlib.Path], tname: Optional[Union[str, pathlib.Path]] = None,
                   ext: str = "png", black_on_white: bool = False):
        if tname is None:
            if self.augmentation_id is not None:
                tname = f"{self.font}_{self.char}_{self.augmentation_id}_{self.unicode}"
            else:
                tname = f"{self.font}_{self.char}_{self.unicode}"

        Glyph.dump_arr(self._sized_img if self._fimg else self.orig_img, tdir, tname, ext, black_on_white)

    def dump_skeleton(self, tdir: Union[str, pathlib.Path], tname: Optional[Union[str, pathlib.Path]] = None,
                      ext: str = "png", black_on_white: bool = False):
        if not self._fskel:
            raise AssertionError("Skeleton was not yet computed. Please invoke 'self.compute_for_size'.")

        if tname is None:
            if self.augmentation_id is not None:
                tname = f"skel_{self.font}_{self.char}_{self.augmentation_id}_{self.unicode}"
            else:
                tname = f"skel_{self.font}_{self.char}_{self.unicode}"
        Glyph.dump_arr(self.skeleton.astype(np.float32), tdir, tname, ext, black_on_white)

    @staticmethod
    def dump_arr(arr: np.ndarray, tdir: Union[str, pathlib.Path],
                 tname: Union[str, pathlib.Path], ext: str = "png", black_on_white: bool = False):
        if black_on_white:
            arr = 1 - arr
        t_image = (arr * 255).astype(np.int32)
        tgt = pathlib.Path(tdir).joinpath(f"{tname}.{ext}")
        cv2.imwrite(tgt.as_posix(), t_image)

    @staticmethod
    def dump_batch(glyphs: List["Glyph"],
                   size: Sequence[int], tdir: Union[str, pathlib.Path],
                   tname: Optional[Union[str, pathlib.Path]] = None, ext: str = "png"):
        raise NotImplementedError
