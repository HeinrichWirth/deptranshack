import ctypes
from ctypes import wintypes


def rss():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(s,ctypes.c_size_t) for s in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    c=Counters();c.cb=ctypes.sizeof(c)
    handle=ctypes.windll.kernel32.GetCurrentProcess;handle.restype=wintypes.HANDLE
    call=ctypes.windll.psapi.GetProcessMemoryInfo;call.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not call(handle(),ctypes.byref(c),c.cb):raise ctypes.WinError()
    return dict(rss_bytes=int(c.WorkingSetSize),peak_rss_bytes=int(c.PeakWorkingSetSize),private_commit_bytes=int(c.PagefileUsage))
