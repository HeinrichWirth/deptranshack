import os,importlib.util
def load():
    p=os.environ.get('REALTIME_NATIVE_LIBRARY')
    if not p:
        from .native import _c4_rt
        return _c4_rt
    spec=importlib.util.spec_from_file_location('_c4_rt',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
