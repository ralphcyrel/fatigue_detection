"""
Optional MJPEG stream of the annotated camera view, served on the LAN.

DIAGNOSTIC VIEW ONLY. The in-vehicle display never shows the driver's face
(group decision, 2026-10-05); this stream exists so the camera view can be
shown on a separate screen - a laptop or projector during the defense, or
the operator's phone while positioning a driver at enrollment. It is off
unless ``main.py --debug-stream`` is given, it serves private-network
clients only, and nothing is recorded: no frame is written to disk and
nothing is sent to the backend. Frames exist only in memory, one at a time.

Access needs a random token in the URL (``http://<pi>:<port>/<token>/``);
every other path is a 404, so someone on the same network who finds the
port still cannot open the view. The token is ``FATIGUE_STREAM_TOKEN`` when
set (a URL that survives restarts - the launcher passes its own to every
session), else a fresh one per run, logged at startup.

It must never slow the detection loop:

* With no viewer connected, :meth:`DebugStream.wants_frame` is False, so the
  loop does not annotate, copy or encode anything.
* With a viewer, the loop hands over at most ``fps`` frames a second
  (:meth:`offer`): one ``frame.copy()`` into a single-slot buffer, replacing
  any frame the encoder has not picked up yet - frames are dropped, the loop
  never waits.
* JPEG encoding runs on its own thread (OpenCV releases the GIL inside
  ``imencode``), and each viewer has its own sender thread that only ever
  sends the newest JPEG, so a slow viewer stalls only itself.
"""

import hmac
import ipaddress
import logging
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

JPEG_QUALITY = 70
BOUNDARY = b"frame"
CLIENT_SEND_TIMEOUT_S = 5.0   # a viewer that stops reading is dropped after this

# Token alphabet: no 0/o, 1/l/i, so it can be read off the panel and typed.
TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
TOKEN_LENGTH = 10             # ~49 bits
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fatigue unit - diagnostic view</title>
<style>body{margin:0;background:#111;color:#bbb;font:14px sans-serif;text-align:center}
img{max-width:100%;max-height:92vh;margin-top:8px}</style></head>
<body><img src="stream.mjpg" alt="camera"><div>Diagnostic view - live only, nothing is recorded</div></body></html>
"""


def new_token() -> str:
    """A random URL token, easy to type."""
    return "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(TOKEN_LENGTH))


def stream_token(env_value: Optional[str]) -> str:
    """``FATIGUE_STREAM_TOKEN`` if it is usable in a URL path, else a new token."""
    if env_value and TOKEN_RE.match(env_value):
        return env_value
    if env_value:
        logger.warning("FATIGUE_STREAM_TOKEN ignored: use 8-64 of A-Z a-z 0-9 _ - "
                       "(a random token is used instead)")
    return new_token()


def lan_address() -> str:
    """Best guess at this unit's LAN IP (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class DebugStream:
    """MJPEG server + encoder thread. Create once, call :meth:`offer` per frame."""

    def __init__(self, port: int, fps: float = 10.0, token: Optional[str] = None) -> None:
        self.port = port
        self.token = token or new_token()
        self.interval = 1.0 / fps
        self._lock = threading.Lock()
        self._new_raw = threading.Condition(self._lock)
        self._new_jpeg = threading.Condition(self._lock)
        self._raw: Optional[np.ndarray] = None   # newest frame not yet encoded
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
        self._viewers = 0
        self._next_due = 0.0
        self._closed = False

        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:   # keep stderr quiet
                logger.debug("debug stream %s: " + fmt, self.client_address[0], *args)

            def do_GET(self) -> None:
                if not ipaddress.ip_address(self.client_address[0]).is_private:
                    self.send_error(403, "LAN only")
                    return
                # /<token>/ and /<token>/stream.mjpg only; anything else 404s
                # (also a wrong token - nothing tells a guesser it was close).
                parts = self.path.split("?", 1)[0].strip("/").split("/")
                if not parts or not hmac.compare_digest(parts[0].encode(),
                                                        stream.token.encode()):
                    self.send_error(404)
                    return
                rest = "/".join(parts[1:])
                if rest == "" and not self.path.split("?", 1)[0].endswith("/"):
                    # Typed without the trailing slash: the page's relative
                    # stream link needs it.
                    self.send_response(301)
                    self.send_header("Location", f"/{stream.token}/")
                    self.end_headers()
                    return
                if rest in ("", "index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE)
                elif rest == "stream.mjpg":
                    stream._serve(self)
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]     # the real port when 0 was asked
        threading.Thread(target=self._server.serve_forever, name="debug-stream-http",
                         daemon=True).start()
        threading.Thread(target=self._encode_loop, name="debug-stream-jpeg",
                         daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://{lan_address()}:{self.port}/{self.token}/"

    # ---- main-loop side (must stay cheap) ------------------------------------

    def wants_frame(self, now: Optional[float] = None) -> bool:
        """
        True when a viewer is connected and a frame is due; claims the slot.

        Called once per loop frame (at capture). The schedule carries over a
        late frame's lateness, so a 24 fps loop still averages ``fps`` rather
        than every third frame (8 fps).
        """
        if self._viewers == 0:
            return False
        now = time.monotonic() if now is None else now
        if now < self._next_due:
            return False
        self._next_due = max(self._next_due, now - self.interval) + self.interval
        return True

    def offer(self, frame: np.ndarray) -> None:
        """Hand one annotated frame to the encoder; never blocks on it."""
        copy = frame.copy()          # the loop reuses nothing, but stay safe
        with self._lock:
            self._raw = copy         # an unencoded older frame is dropped
            self._new_raw.notify()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._new_raw.notify_all()
            self._new_jpeg.notify_all()
        self._server.shutdown()
        self._server.server_close()

    # ---- background threads ---------------------------------------------------

    def _encode_loop(self) -> None:
        while True:
            with self._lock:
                while self._raw is None and not self._closed:
                    self._new_raw.wait()
                if self._closed:
                    return
                raw, self._raw = self._raw, None
            ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            with self._lock:
                self._jpeg = buf.tobytes()
                self._jpeg_seq += 1
                self._new_jpeg.notify_all()

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type",
                            "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
        handler.send_header("Cache-Control", "no-cache, no-store")
        handler.end_headers()
        handler.connection.settimeout(CLIENT_SEND_TIMEOUT_S)
        client = handler.client_address[0]
        with self._lock:
            self._viewers += 1
            seen = self._jpeg_seq
        logger.info("Debug stream: viewer connected from %s (%d watching)", client, self._viewers)
        try:
            while True:
                with self._lock:
                    while self._jpeg_seq == seen and not self._closed:
                        if not self._new_jpeg.wait(timeout=2.0):
                            break          # no frame for 2 s (e.g. recognition stall)
                    if self._closed:
                        return
                    if self._jpeg_seq == seen or self._jpeg is None:
                        continue
                    jpeg, seen = self._jpeg, self._jpeg_seq
                handler.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
        except (OSError, ValueError):
            pass                            # viewer closed the page / timed out
        finally:
            with self._lock:
                self._viewers -= 1
            logger.info("Debug stream: viewer %s left (%d watching)", client, self._viewers)
