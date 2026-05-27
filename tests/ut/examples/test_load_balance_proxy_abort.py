import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from starlette.requests import ClientDisconnect


def load_proxy_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
    spec = importlib.util.spec_from_file_location(f"load_balance_proxy_server_example_{uuid4().hex}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeRequest:
    def __init__(self):
        self.payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }

    async def json(self):
        return self.payload.copy()

    async def body(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeProxyState:
    def __init__(self):
        self.request_num = 0
        self.aborted = []
        self.prefiller_kv_releases = []
        self.decoder_releases = []

    def abort_prefiller_request(self, prefiller_idx, request_id):
        self.aborted.append((prefiller_idx, request_id))

    def release_prefiller_kv(self, prefiller_idx, prefiller_score):
        self.prefiller_kv_releases.append((prefiller_idx, prefiller_score))

    def release_decoder(self, decoder_idx, decoder_score):
        self.decoder_releases.append((decoder_idx, decoder_score))


def configure_proxy(monkeypatch, module, stream_fn):
    proxy_state = FakeProxyState()
    monkeypatch.setattr(module, "proxy_state", proxy_state)
    monkeypatch.setattr(module, "global_args", SimpleNamespace(max_retries=1, retry_delay=0.0), raising=False)

    async def precheck_context_length(api, req_data):
        return None

    async def handle_select_instance(api, req_data, request_length):
        return module.InstanceInfo(
            request_id="req-client-abort",
            prefiller_idx=2,
            prefiller_score=10.0,
            prefiller=SimpleNamespace(url="http://prefiller.example/v1"),
            decoder_idx=3,
            decoder_score=20.0,
            decoder=SimpleNamespace(client=object(), url="http://decoder.example/v1"),
        )

    monkeypatch.setattr(module, "_precheck_context_length", precheck_context_length)
    monkeypatch.setattr(module, "_handle_select_instance", handle_select_instance)
    monkeypatch.setattr(module, "stream_service_response_with_retry", stream_fn)
    return proxy_state


async def collect_response_body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


def test_client_cancel_marks_prefiller_request_aborted(monkeypatch):
    module = load_proxy_module()

    async def cancelled_stream(*args, **kwargs):
        raise asyncio.CancelledError()
        yield b""

    proxy_state = configure_proxy(monkeypatch, module, cancelled_stream)

    async def run_case():
        response = await module._handle_completions("/chat/completions", FakeRequest())
        with pytest.raises(asyncio.CancelledError):
            await collect_response_body(response)

    asyncio.run(run_case())

    assert proxy_state.aborted == [(2, "req-client-abort")]
    assert proxy_state.prefiller_kv_releases == [(2, 10.0)]
    assert proxy_state.decoder_releases == [(3, 20.0)]
    assert proxy_state.request_num == 0


def test_decoder_stream_error_still_marks_prefiller_request_aborted(monkeypatch):
    module = load_proxy_module()

    async def failed_stream(*args, **kwargs):
        raise RuntimeError("decoder stream failed")
        yield b""

    proxy_state = configure_proxy(monkeypatch, module, failed_stream)

    async def run_case():
        response = await module._handle_completions("/chat/completions", FakeRequest())
        chunks = await collect_response_body(response)
        assert chunks == []

    asyncio.run(run_case())

    assert proxy_state.aborted == [(2, "req-client-abort")]
    assert proxy_state.prefiller_kv_releases == [(2, 10.0)]
    assert proxy_state.decoder_releases == [(3, 20.0)]
    assert proxy_state.request_num == 0


def test_asgi_send_disconnect_marks_prefiller_request_aborted(monkeypatch):
    module = load_proxy_module()

    async def stream_with_chunk(*args, **kwargs):
        yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n'
        await asyncio.sleep(3600)

    proxy_state = configure_proxy(monkeypatch, module, stream_with_chunk)

    async def receive():
        await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client disconnected")

    async def run_case():
        response = await module._handle_completions("/chat/completions", FakeRequest())
        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)

    asyncio.run(run_case())

    assert proxy_state.aborted == [(2, "req-client-abort")]
    assert proxy_state.prefiller_kv_releases == [(2, 10.0)]
    assert proxy_state.decoder_releases == [(3, 20.0)]
    assert proxy_state.request_num == 0
