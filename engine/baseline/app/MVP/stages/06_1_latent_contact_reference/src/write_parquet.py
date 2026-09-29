"""Serialize the requested columnar station table using an installed Arrow runtime."""
from lc_common import *
sys.path.insert(0,str(ROOT/'MVP/stages/06_pipeline_performance/vendor'))
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    rows=load(OUT/'per_station.json');table=pa.Table.from_pylist(rows);path=OUT/'per_station.parquet';pq.write_table(table,path,compression='zstd');check=pq.read_table(path);assert check.equals(table);save(OUT/'audit/PARQUET_VERIFIED.json',dict(rows=check.num_rows,columns=check.num_columns,schema=str(check.schema),pyarrow_version=pa.__version__,sha256=sha(path),roundtrip_identical=True));print('PARQUET VERIFIED',check.num_rows,check.num_columns)

if __name__=='__main__':main()
