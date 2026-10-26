Trace: mock4_chat_x0.5 (546 requests), simulated replicas

| policy | TTFT p50 (s) | TTFT p99 (s) | E2E p99 (s) | TPOT p50 (ms) | out tok/s | replica hit rate | router est. hit | max/mean load | errors |
|---|---|---|---|---|---|---|---|---|---|
| round_robin | 0.024 | 0.066 | 1.25 | 4.6 | 1252 | 55.7% | 55.7% | 1.00 | 0 |
| least_load | 0.024 | 0.070 | 1.26 | 4.6 | 1253 | 56.1% | 56.1% | 1.16 | 0 |
| prefix | 0.018 | 0.033 | 1.23 | 4.5 | 1254 | 79.3% | 79.3% | 1.24 | 0 |
| linear | 0.019 | 0.045 | 1.18 | 4.5 | 1253 | 74.8% | 74.8% | 1.04 | 0 |
| lmetric | 0.018 | 0.036 | 1.20 | 4.5 | 1254 | 78.5% | 78.5% | 1.08 | 0 |
