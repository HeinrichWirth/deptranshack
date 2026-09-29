FROM python:3.12.14-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake make && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pybind11==3.0.1
COPY native /native
COPY scripts/build_native.py /build_native.py
RUN python /build_native.py

FROM python:3.12.14-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 C4_BACKEND=full_native_exact PERSISTENT_C4=OFF
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY engine /app/engine
COPY --from=build /compiled/app /app/engine/baseline/app
COPY --from=build /compiled/native /app/engine/native
COPY scripts /app/scripts
RUN python scripts/seal_build.py
COPY app /app/app
COPY web /app/web
COPY tests /app/tests
RUN mkdir /data
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=2)"
CMD ["python", "-m", "app.server"]
