import queue
import threading
import time
import traceback

import numpy as np

from .probe import ProbePlayer


class AudioError(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice as sd
    except OSError as e:
        raise AudioError(f"sounddevice/PortAudio not available: {e}") from e
    except ImportError as e:
        raise AudioError("sounddevice isn't installed (pip install sounddevice)") from e
    return sd


IN_GOOD = {"microphone array": 6, "mic array": 6, "internal": 5, "built-in": 5, "builtin": 5,
           "macbook": 5, "smart sound": 2, "digital mic": 3, "intel": 2, "realtek": 2,
           "microphone": 1, "array": 2, "default": 2, "pipewire": 3, "pulse": 3}
OUT_GOOD = {"speaker": 6, "internal": 5, "built-in": 5, "builtin": 5, "macbook": 5,
            "realtek": 2, "laptop": 2, "default": 2, "pipewire": 3, "pulse": 3}
BAD = {"bluetooth": -14, "hands-free": -14, "handsfree": -14, "headset": -9, "airpods": -14,
       "buds": -12, "rockerz": -12, " bt": -6, "usb": -5, "stereo mix": -25, "what u hear": -25,
       "loopback": -25, "monitor of": -25, "sound mapper": -6, "primary sound": -6,
       "virtual": -15, "cable": -12, "voicemeeter": -15, "steam": -15, "nvidia": -12,
       "hdmi": -14, "displayport": -14, "display audio": -14, "amd high definition": -12,
       "spdif": -12, "s/pdif": -12, "digital output": -8, "webcam": -3, "camera": -3,
       "line in": -6, "headphone": -7, "zoom": -10, "teams": -10, "obs": -10, "iphone": -10,
       "lg ": -10, "dell u": -8, "samsung": -6}
HOSTAPI = {"windows wasapi": 4, "core audio": 4, "mme": 1, "windows directsound": 0,
           "windows wdm-ks": -9, "alsa": 1, "jack audio connection kit": -3, "asio": -5}


def score(dev, hostapi, kind, is_default):
    name = " " + dev["name"].lower() + " "
    s = 0.0
    for k, v in (IN_GOOD if kind == "input" else OUT_GOOD).items():
        if k in name:
            s += v
    for k, v in BAD.items():
        if k in name:
            s += v
    s += HOSTAPI.get(hostapi.lower(), 0)
    if is_default:
        s += 2
    ch = dev["max_input_channels"] if kind == "input" else dev["max_output_channels"]
    if kind == "input" and ch >= 2:
        s += 1
    if kind == "output" and ch < 2:
        s -= 6
    return s


def candidates(kind):
    sd = _sd()
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    try:
        d_in, d_out = sd.default.device
    except Exception:
        d_in = d_out = -1
    out = []
    for i, d in enumerate(devs):
        ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if ch < 1:
            continue
        api = apis[d["hostapi"]]["name"]
        is_def = i == (d_in if kind == "input" else d_out) or i == apis[d["hostapi"]].get(
            "default_input_device" if kind == "input" else "default_output_device", -2)
        out.append((score(d, api, kind, is_def), i, d, api))
    out.sort(key=lambda r: -r[0])
    return out


def describe_devices():
    sd = _sd()
    lines = [f"PortAudio: {sd.get_portaudio_version()[1]}", ""]
    for kind in ("input", "output"):
        lines.append(f"{kind} devices, best first:")
        for sc, i, d, api in candidates(kind):
            ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
            lines.append(f"  [{i:3d}] {sc:+5.1f}  {d['name'][:52]:52s} {api:22s} {ch}ch "
                         f"{int(d['default_samplerate'])} Hz")
        lines.append("")
    return "\n".join(lines)


def _wasapi(sd, raw, auto):
    ws = sd.WasapiSettings(auto_convert=auto)
    if raw:
        try:
            ws._streaminfo.streamOption = sd._lib.eStreamOptionRaw
        except Exception:
            pass
    return ws


class AudioIO:
    def __init__(self, make_engine, in_dev=None, out_dev=None):
        self.make_engine = make_engine
        self.q = queue.Queue(maxsize=400)
        self.stop_evt = threading.Event()
        self.engine = None
        self.player = None
        self.in_stream = None
        self.out_stream = None
        self.skipped_s = 0.0
        self.info = {}
        self._open(in_dev, out_dev)

    def _open(self, in_dev, out_dev):
        sd = _sd()
        ins = candidates("input")
        outs = candidates("output")
        if in_dev is not None:
            ins = [r for r in ins if r[1] == int(in_dev)] or ins
        if out_dev is not None:
            outs = [r for r in outs if r[1] == int(out_dev)] or outs
        if not ins or not outs:
            raise AudioError("no audio input or output devices found")
        tried = []
        for sc_i, i, di, api_i in ins[:6]:
            for sc_o, o, do, api_o in sorted(outs, key=lambda r: (r[3] != api_i, -r[0]))[:4]:
                if sc_i < -5 or sc_o < -5:
                    continue
                rates = [48000]
                for d in (di, do):
                    r = int(d["default_samplerate"])
                    if r >= 44100 and r not in rates:
                        rates.append(r)
                if 44100 not in rates:
                    rates.append(44100)
                modes = ["wasapi-raw", "wasapi"] if "wasapi" in api_i.lower() else ["default"]
                attempts = [(fs, m) for m in modes for fs in rates]
                if "wasapi" in api_i.lower():
                    attempts += [(48000, "wasapi-convert"), (44100, "wasapi-convert")]
                for fs, mode in attempts:
                    try:
                        self._try(sd, i, di, o, do, fs, mode)
                    except Exception as e:
                        tried.append(f"in[{i}] out[{o}] {fs} Hz {mode}: {str(e)[:80]}")
                        self._close_streams()
                        continue
                    self.info = {"in_name": di["name"], "out_name": do["name"]}
                    print(f"in : [{i}] {di['name']} ({api_i}, {self.n_mics} ch, {mode})")
                    print(f"out: [{o}] {do['name']} ({api_o}) @ {fs} Hz")
                    return
        raise AudioError("couldn't open any input/output pair:\n  " + "\n  ".join(tried[-12:]))

    def _try(self, sd, i, di, o, do, fs, mode):
        n_in = 2 if di["max_input_channels"] >= 2 else 1
        extra_in = extra_out = None
        if mode.startswith("wasapi"):
            extra_in = _wasapi(sd, raw=(mode == "wasapi-raw"), auto=(mode == "wasapi-convert"))
            extra_out = _wasapi(sd, raw=False, auto=(mode == "wasapi-convert"))
        self.n_mics = n_in
        self.fs = fs
        self.engine = self.make_engine(fs, n_in)
        table, ver = self.engine.output_table()
        self.player = ProbePlayer(table)
        self._ver = ver
        self.in_stream = sd.InputStream(device=i, channels=n_in, samplerate=fs, dtype="float32",
                                        blocksize=0, latency="low", callback=self._in_cb,
                                        extra_settings=extra_in)
        self.out_stream = sd.OutputStream(device=o, channels=2, samplerate=fs, dtype="float32",
                                          blocksize=0, latency="high", callback=self._out_cb,
                                          extra_settings=extra_out)

    def _in_cb(self, indata, frames, t, status):
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _out_cb(self, outdata, frames, t, status):
        try:
            table, ver = self.engine.output_table()
            if ver != self._ver:
                self._ver = ver
                self.player.request(table)
            outdata[:] = self.player.render(frames)
        except Exception:
            outdata.fill(0)

    def start(self):
        self.engine.info.update(self.info)
        self.out_stream.start()
        self.in_stream.start()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _work(self):
        while not self.stop_evt.is_set():
            try:
                blk = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            blocks = [blk]
            while True:
                try:
                    blocks.append(self.q.get_nowait())
                except queue.Empty:
                    break
            x = np.concatenate(blocks)
            eng = self.engine
            if eng.state == "tracking":
                lag = len(x) - int(0.06 * self.fs)
                if len(x) > 0.25 * self.fs and lag >= eng.N:
                    cut = (lag // eng.N) * eng.N
                    x = x[cut:]
                    self.skipped_s += cut / self.fs
            try:
                eng.feed(x)
            except Exception as e:
                traceback.print_exc()
                eng.error = {"message": f"Signal processing crashed: {e}", "hints": ["press R to restart"]}
                eng.state = "error"

    def stop(self):
        self.stop_evt.set()
        try:
            if self.player:
                self.player.fade_out()
                time.sleep(0.08)
        except Exception:
            pass
        self._close_streams()

    def _close_streams(self):
        for st in (self.in_stream, self.out_stream):
            try:
                if st is not None:
                    st.stop()
                    st.close()
            except Exception:
                pass
        self.in_stream = self.out_stream = None

    def busy(self):
        return self.q.qsize() > 6
