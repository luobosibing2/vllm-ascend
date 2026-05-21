# Load Balance Proxy Length Precheck

## Goal

`load_balance_proxy_server_example.py` now validates request length before it dispatches a request to the prefiller and decoder nodes. This prevents requests that exceed a node's model context length from reaching the backend after the proxy has already started the prefill/decode flow.

This is especially useful when prefiller and decoder nodes use different `--max-model-len` values:

- the prefiller node only needs the rendered prompt to fit in the prefiller context;
- the decoder node needs the rendered prompt plus the requested output budget to fit in the decoder context.

## Plan

The proxy performs a lightweight precheck before `_handle_select_instance()`:

1. Build a `/tokenize` payload from the original client request.
2. Call `/tokenize` on one available prefiller node.
3. Reject the request if the prefiller prompt token count is greater than or equal to the prefiller `max_model_len`.
4. Call `/tokenize` on one available decoder node.
5. Reject the request if the decoder prompt token count is greater than or equal to the decoder `max_model_len`.
6. Reject the request if `decoder_prompt_tokens + max_output_tokens` is greater than the decoder `max_model_len`.
7. If the precheck succeeds, continue through the original prefiller-first scheduling path.

The proxy does not select the actual prefiller or decoder during precheck. It uses representative nodes because all prefiller nodes are expected to share the same model, tokenizer, chat template, and `max_model_len`, and all decoder nodes are expected to do the same within the decoder group.

## Implementation Notes

The vLLM `/tokenize` endpoint is served from the backend root path, for example:

```bash
curl -X POST http://localhost:8100/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model":"model_path","messages":[{"role":"user","content":"hello"}]}'
```

It is not served from `/v1/tokenize`.

For chat requests, the proxy forwards the original `messages` and chat-template-related fields to `/tokenize`. The backend therefore counts tokens after applying its own `chat_template.jinja`, instead of relying on byte-length estimation or manual prompt reconstruction in the proxy.

For completion requests, this precheck currently covers `prompt: str`.

## Output Token Budget

For `/v1/chat/completions`, the decoder check uses:

1. `max_completion_tokens`, when it is present;
2. otherwise `max_tokens`;
3. otherwise no explicit output-length sum check.

For `/v1/completions`, the decoder check uses `max_tokens`, defaulting to vLLM's completion default of `16` when the request omits it.

## Timeout And Failover

`/tokenize` calls use `--tokenize-timeout`, which defaults to `5` seconds:

```bash
python load_balance_proxy_server_example.py \
  --host localhost \
  --prefiller-hosts host1 host2 \
  --prefiller-ports 8100 8101 \
  --decoder-hosts host3 host4 \
  --decoder-ports 8200 8201 \
  --tokenize-timeout 5
```

If `/tokenize` fails or times out on one representative node, the proxy tries the next available node of the same type. If all prefiller `/tokenize` requests fail, the proxy returns `503`. If all decoder `/tokenize` requests fail, the proxy also returns `503`.

## Expected Behavior

- If the prompt is too long for the prefiller context, the proxy returns `400` before sending any prefill request.
- If the prompt is too long for the decoder context, the proxy returns `400` before sending any prefill request.
- If `prompt_tokens + max_output_tokens` is too long for the decoder context, the proxy returns `400` before sending any prefill request.
- If `/tokenize` is unavailable for all nodes in the required node group, the proxy returns `503`.
- If the precheck passes, the request continues through the existing P-first scheduling, KV transfer, and decode path.

## Validation

Suggested validation cases:

1. Send a chat request whose rendered prompt length is greater than or equal to the prefiller `max_model_len`; expect `400`.
2. Send a request where the decoder prompt fits but `prompt_tokens + max_tokens` exceeds the decoder `max_model_len`; expect `400`.
3. Send a boundary request where `prompt_tokens + max_tokens == decoder max_model_len`; expect the request to pass precheck.
4. Stop or hang the first prefiller `/tokenize` endpoint while another prefiller is healthy; expect the proxy to try the next prefiller.
5. Stop all decoder `/tokenize` endpoints; expect `503`.

## Boundaries

This precheck is a targeted hardening for the example load-balance proxy. It does not replace production-grade health checks, circuit breaking, service discovery, backend draining, or router-level observability.
