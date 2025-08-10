from .template import SkelAlgorithm

from .zhang import SkelZhang
from .lee import SkelLee
from .medial_axis import SkelMedialAxis
from .aof import SkelAOF


def get_skel_method(method_name: str):
    if method_name.casefold() == "zhang".casefold():
        return SkelZhang
    elif method_name.casefold() == "lee".casefold():
        return SkelLee
    elif method_name.casefold() == "medial_axis".casefold():
        return SkelMedialAxis
    elif method_name.casefold() == "aof".casefold():
        return SkelAOF
    else:
        raise ValueError(f"Provided method name '{method_name}' does not match with any implementations.")