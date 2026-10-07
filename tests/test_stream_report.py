"""Portal live view: the stream's address rides on the heartbeat and is withdrawn at shutdown."""

import logging

import requests

from config import config
from modules.api import STREAM_CLEAR_TIMEOUT_S, APIClient
from modules.debug_stream import DebugStream


class Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload
        self.text = "" if payload is None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


class Session:
    """Stands in for requests.Session: records calls, answers from a script."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        answer = self.answers.pop(0) if self.answers else Resp(200, {"status": "ok"})
        if isinstance(answer, Exception):
            raise answer
        return answer


def client(*answers):
    c = APIClient(base_url="http://portal.test/api", token="t")
    c.session = Session(*answers)
    return c


class RecordingAPI:
    def __init__(self):
        self.calls = []

    def post_heartbeat(self, relay):
        self.calls.append(("post_heartbeat",))

    def report_stream(self, lan_ip, port, token):
        self.calls.append(("report_stream", lan_ip, port, token))

    def clear_stream(self):
        self.calls.append(("clear_stream",))


# ---- APIClient -------------------------------------------------------------

def test_report_sends_only_the_address_parts():
    c = client(Resp(200, {"status": "ok", "expires_in": 90}))
    assert c.report_stream("192.168.1.42", 8080, "testtoken1", blocking=True)
    method, url, kw = c.session.calls[0]
    assert (method, url) == ("PUT", f"http://portal.test/api/devices/{config.DEVICE_ID}/stream")
    assert kw["json"] == {"lan_ip": "192.168.1.42", "port": 8080, "token": "testtoken1"}
    assert kw["timeout"] == c.timeout


def test_report_logs_changes_only(caplog):
    c = client(Resp(200, {"status": "ok"}), Resp(200, {"status": "ok"}),
               Resp(422, {"message": "lan_ip"}), Resp(422, {"message": "lan_ip"}),
               requests.exceptions.ConnectionError("down"), Resp(200, {"status": "disabled"}))
    with caplog.at_level(logging.INFO, logger="modules.api"):
        results = [c.report_stream("10.0.0.5", 8080, "testtoken1", blocking=True) for _ in range(6)]
    assert results == [True, True, False, False, False, True]
    messages = [r.getMessage() for r in caplog.records if r.name == "modules.api"]
    assert sum("listed on the portal" in m for m in messages) == 1
    assert sum("rejected" in m for m in messages) == 1
    assert sum("live view is switched off" in m for m in messages) == 1


def test_clear_is_short_and_never_raises():
    c = client(Resp(200, {"status": "cleared"}))
    assert c.clear_stream()
    method, url, kw = c.session.calls[0]
    assert (method, url) == ("DELETE", f"http://portal.test/api/devices/{config.DEVICE_ID}/stream")
    assert kw["timeout"] == STREAM_CLEAR_TIMEOUT_S

    offline = client(requests.exceptions.ConnectTimeout("slow"))
    assert offline.clear_stream() is False


# ---- Heartbeat / cleanup in main.py ------------------------------------------

def test_heartbeat_without_stream_reports_nothing_extra(m):
    api = RecordingAPI()
    hb = m.Heartbeat(api, None, interval=30)
    hb.tick(now=0.0)
    hb.withdraw_stream()
    assert api.calls == [("post_heartbeat",)]


def test_stream_is_reported_with_every_beat_and_withdrawn_at_cleanup(m, monkeypatch):
    monkeypatch.setattr(m, "lan_address", lambda: "192.168.1.42")
    m._stream = DebugStream(0, 10, "testtoken1")
    api = RecordingAPI()
    m._heartbeat = m.Heartbeat(api, None, interval=30, stream=m._stream)

    assert m._heartbeat.tick(now=0.0)
    assert not m._heartbeat.tick(now=10.0)          # between beats: nothing
    assert m._heartbeat.tick(now=30.0)
    port = m._stream.port
    assert api.calls == [
        ("post_heartbeat",), ("report_stream", "192.168.1.42", port, "testtoken1"),
        ("post_heartbeat",), ("report_stream", "192.168.1.42", port, "testtoken1"),
    ]

    m.cleanup()
    assert api.calls[-1] == ("clear_stream",)
    m._stream = None                                 # cleanup() already closed it


def test_cleanup_does_not_clear_a_stream_that_was_never_listed(m):
    m._stream = DebugStream(0, 10, "testtoken1")
    api = RecordingAPI()
    m._heartbeat = m.Heartbeat(api, None, interval=30, stream=m._stream)
    m.cleanup()
    assert api.calls == []
    m._stream = None
