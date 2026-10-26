.PHONY: install test bench-mock bench-mock-agent up-mock up-gpu down

install:
	pip install -e ".[test,plot,hf]"

test:
	pytest -q

bench-mock:        ## policy comparison on 4 simulated replicas, chat trace
	python -m kvroute.bench --mock 4 --kind chat --scale 0.5 --rate-scale 2

bench-mock-agent:  ## harder case: long shared agent prefixes, slower prefill, small cache
	python -m kvroute.bench --mock 4 --kind agent --rate-scale 3 \
		--mock-args "--prefill-ms-per-token 0.15 --decode-ms-per-seq 0.3 --num-gpu-blocks 1024"

up-mock:           ## router + 4 simulated replicas + Prometheus + Grafana (no GPU)
	cd deploy && docker compose -f docker-compose.mock.yml up -d --build

up-gpu:            ## router + 4 vLLM replicas + Prometheus + Grafana
	cd deploy && docker compose up -d --build

down:
	cd deploy && docker compose down; docker compose -f docker-compose.mock.yml down
