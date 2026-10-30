# Recorded loopback protocol checks

`contract.json` records actual HTTP/NDJSON results against the random neural
CPU reference, together with source and dependency hashes/versions. It is not
a benchmark: the checker injects 10 ms per neural call to make incremental
delivery observable. Actual generated IDs are compared with uncached greedy.

All servers use OS-assigned `127.0.0.1` ports and are stopped by the checker.
The normal stream delivers a token before the engine's terminal time. The
record also includes cancellations, timeout-versus-completion, disconnect
cleanup, zeroed ownership maps, a model-call error, a fatal owner-loop failure
and failed/timed-out startup. The response-start failure is a separate in-process
ASGI injection, not a claimed network reproduction of that exact send boundary.

Reproduce with the command in practice 14a. The program prints a fresh JSON to
stdout and its completion result to stderr; it does not overwrite this snapshot.
The reference is a custom token-ID protocol, not the original course client's
text-prompt API or an OpenAI-compatible server. Checkpoint/tokenizer, container,
GPU, authentication and production-capacity work remain separate.
