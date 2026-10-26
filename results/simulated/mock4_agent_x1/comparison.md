Trace: mock4_agent_x1 (908 requests), simulated replicas

| policy | TTFT p50 (s) | TTFT p99 (s) | E2E p99 (s) | TPOT p50 (ms) | out tok/s | replica hit rate | router est. hit | max/mean load | errors |
|---|---|---|---|---|---|---|---|---|---|
| round_robin | 0.173 | 0.866 | 6.35 | 21.6 | 1276 | 66.4% | 66.5% | 1.00 | 0 |
| least_load | 0.198 | 1.000 | 4.92 | 25.9 | 1274 | 66.3% | 66.4% | 1.04 | 0 |
| prefix | 0.092 | 1.142 | 4.16 | 13.2 | 1481 | 78.4% | 78.5% | 1.13 | 0 |
| linear | 0.095 | 1.083 | 4.17 | 13.0 | 1436 | 77.5% | 77.3% | 1.02 | 0 |
| lmetric | 0.064 | 1.199 | 3.79 | 11.3 | 1500 | 81.1% | 80.6% | 1.08 | 0 |
