import torch.distributed as dist
import itertools


def is_dist():
    return dist.is_initialized()

def is_main_process():
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_zero_only(func):
    """Decorator to make a function only run on the main process."""

    def wrapper(*args, **kwargs):
        """Wrapper function for the decorator."""
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return wrapper


def repeat_texts(
    texts,
    num_repeats):
    return list(itertools.chain(*[[s for _ in range(num_repeats)]
                                    for s in texts]))