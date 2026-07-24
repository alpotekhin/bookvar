---
title: "Week07 Application Deployment — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week07_application_deployment/Effdl26-07.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment/Effdl26-07.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment/Effdl26-07.pdf?raw=1" title="Week07 Application Deployment — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
Efficient DL 2026: Part 07
TCP/IP, HTTP, Docker, Orchestration, Deployment Basics
```

### Page 2

```text
Today
●   Intro to TCP/IP & HTTP
●   Flask web server deployment
●   Metrics logging with Prometheus and Grafana
●   Intro to gRPC and Protobuf


In your homework…
●   uv+pytest
●   ML classifier service with HTTP and gRPC
●   Prometheus service
```

### Page 3

```text
4 layers of TCP/IP
●   Application layer: protocols e.g. HTTP, HTTPS, SMTP, SSH etc.
●   Transport layer: e.g. TCP or UDP
●   Internet layer: IP addresses, routing, name resolution, subnets etc.
●   Network Interface layer: physical devices, electrical signal processing
```

### Page 4

```text
Playground
```

### Page 5

```text
HTTP Requests
Each HTTP request contains

●   First line (Method, path, http version etc.)
●   Headers (address, data format specifications, user-agent
●   Body
```

### Page 6

```text
HTTP responses
```

### Page 7

```text
Playground
```

### Page 8

```text
Python http server – Flask
Pros:
●   Simple and straghtforward
●   Written in python
●   Can run (almost) any python code
    (including ML models!)
Cons:
●   Written in python
●   Not secure
●   Does not scale well
```

### Page 9

```text
How to scale up your deployment?
●   Host machine must be very powerful
●   Good network connectivity is required
●
```

### Page 10

```text
How to scale up your deployment?
●   Host machine must be very powerful
●   Good network connectivity is required
●   Create several instances with load balancer!
```

### Page 11

```text
Playground
```

### Page 12

```text
Docker
●   Is used to create isolated environments within
    your OS
```

### Page 13

```text
Playground
```

### Page 14

```text
Orchestration

Deploy docker images to multiple machines
```

### Page 15

```text
Playground
```
