import bz2
import pickle
import click


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Glyph':
            from glyph import Glyph
            return Glyph
        return super().find_class(module, name)


def compressed_pickle(data, save_path):
    with bz2.BZ2File(save_path, 'wb') as f:
        pickle.dump(data, f)


def decompress_pickle(file):
    with bz2.BZ2File(file, 'rb') as f:
        data = CustomUnpickler(f).load()
    return data


def to_pickle(data, save_path):
    with open(save_path + '.pkl', 'wb') as f:
        pickle.dump(data, f)


def from_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def validate_positive_int(ctx, param, value):
    """Validate that the input is a valid integer."""
    if value <= 0:
        raise click.BadParameter("The provided value must be a positive integer.")
    return value
