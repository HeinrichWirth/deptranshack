"""Separate processes enforce the predict -> evaluate -> diagnose ordering."""
from fusion_common import *
import subprocess
if __name__=='__main__':
    workers=sys.argv[1] if len(sys.argv)>1 else '10'
    for script,args in [('study.py',['heldout']),('assess.py',['heldout']),('diagnose.py',[])]:
        subprocess.run([sys.executable,'-B',str(STAGE/'src'/script),*args,'--workers',workers],check=True)
