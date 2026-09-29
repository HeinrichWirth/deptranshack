from . import ROOT,OUT
import long_common as lc
import laspy


def main():
    run='roundT_squareT_pressureGate_squareT';rows=lc.frames(run)
    target=OUT/'docker_fixture';target.mkdir(exist_ok=True)
    # One actual cloud and two sanitized pose records; T+1 cloud intentionally absent.
    from fusion import sanitized
    lc.save(target/'fixture.frames.json',dict(frames=[sanitized(r) for r in rows[:2]]))
    original=laspy.read(lc.dataset()/run/rows[0]['file']);original.classification[:]=0
    original.write(target/rows[0]['file'])
    (OUT/'docker_test').mkdir(exist_ok=True)


if __name__=='__main__':main()
