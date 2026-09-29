"""Compare transport decoding to the existing rosbags decoder on real messages."""
import sys,sqlite3,json,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import cloud
import numpy as np
from cdr_cloud import decode

def main():
    db=ROOT/'datas/test_synthetic/cloud_with_fake_obj/cloud_with_fake_obj_0.db3'
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    rows=con.execute('SELECT id FROM messages ORDER BY timestamp,id').fetchall()
    for i in (0,1,218,219,230,231,232,369,370,1000,1509):
        data=con.execute('SELECT data FROM messages WHERE id=?',rows[i]).fetchone()[0]
        original,expected=cloud.decode(data);meta,actual=decode(data)
        assert actual.dtype==expected.dtype and actual.shape==expected.shape
        assert actual.tobytes()==expected.tobytes()
        assert meta['header_time_ns']==cloud.stamp_ns(original) and meta['frame_id']==original.header.frame_id
        assert not actual.flags.writeable
    for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf-8-sig'))
    print('ADAPTER_PASS: 11 real messages, all point fields/rows/header stamps exactly match existing rosbags decoder')
    out=ROOT/'results_ros2_replay_v1';out.mkdir(exist_ok=True)
    (out/'adapter_tests.json').write_text(json.dumps(dict(pass_=True,cases=11,decoder='exact match with existing tools/cloud.py rosbags'),indent=2))

if __name__=='__main__':main()
