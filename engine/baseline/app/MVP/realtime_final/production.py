"""Locked production defaults, separate from benchmark entry points."""
SOLVE_WORKERS=2
CANDIDATE_THREADS=2

def create_scheduler(run_directory):
    from .live import Scheduler
    return Scheduler(run_directory,workers=SOLVE_WORKERS,threads=CANDIDATE_THREADS)

def main():
    import sys
    from .cli import main as execute
    if '--threads' not in sys.argv:sys.argv.extend(['--threads',str(CANDIDATE_THREADS)])
    if '--workers' not in sys.argv:sys.argv.extend(['--workers',str(SOLVE_WORKERS)])
    execute()

if __name__=='__main__':main()
