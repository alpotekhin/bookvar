# Executed source checks

`source-checks.json` records the exact unchanged source and checker hashes, selected runtime versions and observed behavior. `source-checks.pytest.txt` retains the original source-suite failure diagnostic (four passes, one known inconsistent expected value). This is an in-process service-logic/instrumentation check with a stubbed VGG model; not Docker, gRPC, Prometheus scraping, model inference, GPU execution or a load benchmark. The invocation is visible in practice 17.
