from contextlib import contextmanager
import time


@contextmanager
def timer(bag: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bag[key] = bag.get(key, 0.0) + (time.perf_counter() - t0)
