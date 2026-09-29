param([string[]]$Arguments, [string]$Log = "", [string]$Data = "C:\Users\heinrich.wirth\deptrans\output\LAS_FRAMES_CLASSIFIED_20260926\ANNOTATED")
$ErrorActionPreference = 'Stop'
$rtWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$rtArgs = @('run','--rm','--entrypoint','python',
 '--mount',"type=bind,source=$rtWorkspace\MVP\realtime_final,target=/app/MVP/realtime_final,readonly",
 '--mount',"type=bind,source=$rtWorkspace\MVP\full_native_final,target=/app/MVP/full_native_final,readonly",
 '--mount',"type=bind,source=$rtWorkspace\results_realtime_final,target=/app/results_realtime_final",
 '--mount',"type=bind,source=$rtWorkspace\results_performance_final2\benchmark_cohort.json,target=/app/results_performance_final2/benchmark_cohort.json,readonly",
 '--mount',"type=bind,source=$Data,target=/data,readonly",'-e','REALTIME_DATA=/data','deptrans-realtime:exact-gcc','-B') + $Arguments
if ($Log) { & docker @rtArgs *> $Log } else { & docker @rtArgs }
if ($LASTEXITCODE -ne 0) { throw "Linux run failed: $LASTEXITCODE" }
