FROM deptrans-realtime:exact-gcc
COPY . /app/MVP/realtime_final
ENV C4_BACKEND=full_native_exact PERSISTENT_C4=OFF OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENTRYPOINT ["python", "-B", "-m", "MVP.realtime_final.production"]
