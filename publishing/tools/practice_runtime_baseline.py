"""Reconstruct the 26 October practice only for historical preservation checks."""
import re

OLD_DEPLOYMENT_BOUNDARY = """No container or Prometheus/Grafana stack was started for this local check:
Docker daemon is unavailable. Container builds, real scrapes, gRPC requests and
a dashboard-backed load experiment remain unverified. The next experiment must
separate offered arrivals, accepted requests, errors, queue/preprocessing/model/
serialization times and client completion. Compare cold start with steady
state, increase a fixed arrival rate until the chosen SLO fails, and retain the
full event records rather than inferring capacity from a successful smoke test."""


def before_observable_runtime(text):
    text, count = re.subn(
        r'The 26 October check above did not start containers.*?'
        r'<!-- END REVIEW: observable runtime 2026-10-29 -->',
        lambda _: OLD_DEPLOYMENT_BOUNDARY, text, flags=re.S)
    if count != 1:
        raise ValueError('Expected exactly one bounded observable-runtime addition')
    return text.replace('last_updated: 2026-10-29', 'last_updated: 2026-10-26').replace(
        'last_verified: 2026-10-29', 'last_verified: 2026-10-26')
