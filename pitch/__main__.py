import argparse
import sys

from .config import Config


def main():
    ap = argparse.ArgumentParser(prog="pitch")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--in", dest="in_dev", type=int)
    ap.add_argument("--out", dest="out_dev", type=int)
    ap.add_argument("--quiet", action="store_true", help="keep the probe above 15 kHz")
    a = ap.parse_args()

    from .audio import AudioError, AudioIO, describe_devices
    if a.list_devices:
        print(describe_devices())
        return 0

    cfg = Config()
    if a.quiet:
        cfg.band_floor = 15000.0

    from .engine import Engine
    try:
        io = AudioIO(lambda fs, mics: Engine(cfg, fs, mics), a.in_dev, a.out_dev)
    except AudioError as e:
        print(f"\nCouldn't open the laptop's audio: {e}\n")
        print("try  python -m pitch --list-devices  and then  --in N --out M")
        return 2

    sys.setswitchinterval(0.002)
    from .ui import App
    app = App(io, cfg)
    io.start()
    try:
        app.run()
    finally:
        io.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
