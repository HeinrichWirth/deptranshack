"""Instance-local execution optimization; frozen source and global NumPy untouched."""
import types
import numpy as np

class NumpyProxy:
    def __getattr__(self, name):
        return getattr(np, name)

    def cross(self, a, b, *args, **kwargs):
        if (not args and not kwargs and isinstance(a, np.ndarray)
                and isinstance(b, np.ndarray) and a.shape == b.shape == (3,)
                and a.dtype == b.dtype == np.dtype('float64')):
            return np.array((a[1]*b[2]-a[2]*b[1],
                             a[2]*b[0]-a[0]*b[2],
                             a[0]*b[1]-a[1]*b[0]))
        return np.cross(a, b, *args, **kwargs)

def install(engine):
    proxy = NumpyProxy()
    memo = {}
    def clone(fn):
        if fn in memo:
            return memo[fn]
        env = dict(fn.__globals__)
        copied = types.FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__, fn.__closure__)
        memo[fn] = copied
        copied.__kwdefaults__ = fn.__kwdefaults__
        for key, value in list(env.items()):
            if value is np:
                env[key] = proxy
            elif isinstance(value, types.FunctionType) and '/baseline/app/' in value.__code__.co_filename.replace('\\', '/'):
                env[key] = clone(value)
        return copied
    engine.process_frame = types.MethodType(clone(engine.process_frame.__func__), engine)
    return len(memo)
