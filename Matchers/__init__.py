from .FrameByFrameMatcher import FrameByFrameMatcher
from .SuperGlueMatcher import SuperGlueMatcher


def create_matcher(conf):
    try:
        code_line = f"{conf['name']}(conf)"
        matcher = eval(code_line)
    except NameError:
        raise NotImplementedError(f"{conf['name']} is not implemented yet.")

    return matcher
