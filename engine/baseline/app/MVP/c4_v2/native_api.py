def startup():
    try:
        from .native import _c4_v2
    except ImportError as e:
        raise RuntimeError('C4_BACKEND=v2 requires _c4_v2. Build MVP/c4_v2/build_native.ps1; no silent reference fallback.') from e
    return _c4_v2
