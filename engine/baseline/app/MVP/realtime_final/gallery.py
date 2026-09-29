"""Reuse the established report format with final Linux outputs."""
from . import OUT
from MVP.full_native_final import gallery as old
def main():
    old.OUT=OUT/'linux_final';old.main()
    p=old.OUT/'gallery.html';s=p.read_text(encoding='utf-8').replace('REPORT_FULL_NATIVE_C4_FINAL.html','../REPORT_REALTIME_FINAL.html').replace('Full native C4 from scratch — квалификация','Final realtime — Linux quality и точная оптимизация')
    p.write_text(s,encoding='utf-8')
if __name__=='__main__':main()
