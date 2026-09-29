"""Start a separate loopback viewer without touching existing servers."""
import json,subprocess,sys,socket
from pathlib import Path
root=Path(__file__).resolve().parent.parent
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1',8768))==0:raise SystemExit('Port 8768 already occupied; inspect instead of replacing')
logs=root/'COPY_MAIN/results/viewer';logs.mkdir(exist_ok=True)
with (logs/'viewer.stdout.log').open('ab') as out,(logs/'viewer.stderr.log').open('ab') as err:
    process=subprocess.Popen([sys.executable,'-B','-X','faulthandler','-u',str(root/'tools/serve.py'),'--data-root',str(root),'--port','8768'],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
(logs/'VIEWER_SERVER.json').write_text(json.dumps(dict(pid=process.pid,port=8768,url='http://127.0.0.1:8768/predicted_train.html'),indent=2)+'\n')
print('STARTED',process.pid)
