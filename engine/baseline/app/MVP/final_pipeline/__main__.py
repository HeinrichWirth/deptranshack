import argparse
from . import process_run

parser = argparse.ArgumentParser(description='Frozen pipeline with authorized RAW seed adapter and one-pose delay.')
parser.add_argument('--run', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--mode', choices=('online', 'research'), default='online')
parser.add_argument('--indices', type=int, nargs='+')
parser.add_argument('--pose-policy', choices=('strict_current_past', 'frozen_delayed_pose_audit'), default='frozen_delayed_pose_audit')
args = parser.parse_args()
config = vars(args).copy()
if config['indices'] is None:
    del config['indices']
process_run(args.run, config)
