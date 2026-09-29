FROM python:3.12.14-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make clang && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pybind11==3.0.1
ARG CXX_COMPILER=g++
COPY . /build/native
RUN cmake -S /build/native -B /build/compiled -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
 && cmake --build /build/compiled --parallel 2 && cmake --install /build/compiled --prefix /install \
 && ${CXX_COMPILER} --version > /install/realtime_compiler.txt
FROM deptrans-c4-v2:experimental AS runtime
COPY --from=builder /install /app
ENV C4_BACKEND=reference PERSISTENT_C4=OFF
ENTRYPOINT ["python","-B","-m","MVP.realtime_final.cli"]
