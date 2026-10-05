"""Diagnostic stream: token-only access, frame pacing, delivery to a viewer."""

import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from modules.debug_stream import TOKEN_ALPHABET, TOKEN_LENGTH, DebugStream, new_token, stream_token


@pytest.fixture
def stream():
    s = DebugStream(0, 10, "testtoken1")
    yield s
    s.close()


def get(s, path):
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{s.port}{path}", timeout=3)
        return resp.status, resp.geturl().split(str(s.port), 1)[1]
    except urllib.error.HTTPError as exc:
        return exc.code, None


def test_only_the_token_path_is_served(stream):
    for path in ("/", "/index.html", "/stream.mjpg", "/wrongtoken9/", "/testtoken1x/"):
        assert get(stream, path)[0] == 404, path
    assert get(stream, "/testtoken1/") == (200, "/testtoken1/")
    assert get(stream, "/testtoken1") == (200, "/testtoken1/"), "missing slash redirects"
    assert stream.url.endswith(f":{stream.port}/testtoken1/")


def test_token_generation_and_env_validation():
    t = new_token()
    assert len(t) == TOKEN_LENGTH and set(t) <= set(TOKEN_ALPHABET)
    assert stream_token("my-fixed-token1") == "my-fixed-token1"
    assert stream_token("short") != "short", "too short -> random token"
    assert stream_token("has space!") != "has space!"
    assert len(stream_token(None)) == TOKEN_LENGTH


def test_no_viewer_means_no_work(stream):
    assert not any(stream.wants_frame(now=t / 30) for t in range(300))


def test_pacing_averages_target_rate(stream):
    stream._viewers = 1
    frames = [i * 0.041 for i in range(int(10 / 0.041))]      # a 24 fps loop for 10 s
    claimed = sum(stream.wants_frame(now=1000 + t) for t in frames)
    assert 95 <= claimed <= 105, f"~10 fps, got {claimed / 10:.1f}"


def test_viewer_receives_frames(stream):
    got = {"n": 0}

    def viewer():
        resp = urllib.request.urlopen(f"http://127.0.0.1:{stream.port}/testtoken1/stream.mjpg",
                                      timeout=5)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            if resp.readline().startswith(b"Content-Length:"):
                got["n"] += 1

    thread = threading.Thread(target=viewer, daemon=True)
    thread.start()
    frame = np.full((480, 640, 3), 90, np.uint8)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.5:
        if stream.wants_frame():
            stream.offer(frame)
        time.sleep(1 / 30)
    thread.join(timeout=5)
    assert got["n"] >= 10, f"viewer got {got['n']} frames in 2 s"
