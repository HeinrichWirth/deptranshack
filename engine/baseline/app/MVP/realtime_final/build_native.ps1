$ErrorActionPreference = 'Stop'
$stageRoot = $PSScriptRoot
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$pythonExe = 'C:\Users\heinrich.wirth\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $cmakeExe -S "$stageRoot/native" -B "$stageRoot/native/build" -G 'Visual Studio 17 2022' -A x64 "-DPython_EXECUTABLE=$pythonExe" "-Dpybind11_DIR=$stageRoot/../performance_final/vendor/pybind11/share/cmake/pybind11"
if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
& $cmakeExe --build "$stageRoot/native/build" --config Release
if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
Copy-Item -LiteralPath "$stageRoot/native/build/Release/_c4_rt.cp312-win_amd64.pyd" -Destination "$stageRoot/native/_c4_rt.cp312-win_amd64.pyd" -Force
