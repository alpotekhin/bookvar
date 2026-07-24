---
title: "Metrics"
type: external-resource
status: imported-source
language: original
source_kind: readme
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week07_application_deployment/03_metrics/README.md`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment/03_metrics/README.md) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.

Put `vgg16.pt` inside this directory

Run 

```bash
docker-compose up --build
```

Visit
* `http://localhost:8080/metrics` - raw metrics from app
* `http://localhost:3000/` or `http://localhost:9090` - login with `admin`/`admin` - grafana to draw metrics

## Further reading

* Prometheus - https://prometheus.io/
* Prometheus & Flask - https://pypi.org/project/prometheus-flask-exporter/
* Grafana - https://grafana.com/
* Telegraf - https://www.influxdata.com/time-series-platform/telegraf/
