PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
docker run --rm \
    -v $(pwd):/io \
    -v /home/duxin/.pypirc:/root/.pypirc \
    ghcr.io/pyo3/maturin publish \
        --manifest-path ./bindings/python/Cargo.toml \
        --skip-existing \
        -i python3.8 -i python3.9 -i python3.10 -i python3.11 -i python3.12 -i python3.13 -i pypy3.8 -i pypy3.9 \
        -r pypi


