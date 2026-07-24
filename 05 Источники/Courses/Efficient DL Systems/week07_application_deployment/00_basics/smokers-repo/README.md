---
title: "week07_application_deployment/00_basics/smokers-repo/README.md"
type: external-resource
status: imported-source
language: original
source_kind: readme
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week07_application_deployment/00_basics/smokers-repo/README.md`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment/00_basics/smokers-repo/README.md) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.

## Local run

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Docker run

```bash
docker build -f Dockerfile.bad -t smoker-demo .
docker run --rm -p 8080:8080 smoker-demo
```

## Endpoint

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"you are an idiot"}'
```
