import cProfile
from io import StringIO
import pstats


def using_cprofile(func, root_abs_path=None, replace_pattern=None, save_path=None):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        ret = func(*args, **kwargs)
        pr.disable()
        profile_stats = StringIO()
        stats = pstats.Stats(pr, stream=profile_stats)
        if root_abs_path is not None:
            stats.sort_stats('cumulative').print_stats(root_abs_path)
        else:
            stats.sort_stats('cumulative').print_stats()
        output = profile_stats.getvalue()
        if replace_pattern is not None:
            output = output.replace(replace_pattern, "")
        if save_path is None:
            print(output)
        else:
            with open(save_path, "w") as f:
                f.write(output)
        return ret

    return inner
