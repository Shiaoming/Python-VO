from .KITTILoader import KITTILoader
from .SequenceImageLoader import SequenceImageLoader
from .TUMRGBLoader import TUMRGBLoader


def create_dataloader(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        loader = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return loader
