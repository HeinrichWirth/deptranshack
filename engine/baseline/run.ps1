param([string]$RunName = ('run_' + (Get-Date -Format 'yyyyMMdd_HHmmss')), [int]$Count = 1510, [int]$SolveStride = 1, [int]$SolveOffset = 0, [ValidateSet(1,2)][int]$InputStride = 1, [switch]$ImmediateDirection = $true, [switch]$AsyncDirection = $true)
$ErrorActionPreference = 'Stop'
$copyRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$workspaceRoot = Split-Path $copyRoot -Parent
if ($RunName -notmatch '^[a-zA-Z0-9_]+$') { throw 'RunName must contain letters, digits or underscores' }
if ($SolveStride -lt 1 -or $SolveOffset -lt 0 -or $SolveOffset -ge $SolveStride) { throw 'Invalid solve stride/offset' }
if (Test-Path -LiteralPath (Join-Path $copyRoot "results\$RunName")) { throw 'Choose a new result name' }
$containerName = 'copy-main-' + [guid]::NewGuid().ToString('N').Substring(0,12)
$argsDocker = @('run','--rm','--name',$containerName,'--network','none','--entrypoint','python',
    '--mount',"type=bind,source=$copyRoot,target=/work",
    '--mount',"type=bind,source=$copyRoot\app,target=/candidate,readonly",
    '--mount',"type=bind,source=$workspaceRoot\datas\test_synthetic\cloud_with_fake_obj,target=/source,readonly",
    '--mount',"type=bind,source=$workspaceRoot\MVP\realtime_final,target=/freeze,readonly",
    'deptrans-realtime:production','-B','-u','/work/run_once.py',$RunName,'--count',"$Count",'--solve-stride',"$SolveStride",'--solve-offset',"$SolveOffset",'--input-stride',"$InputStride")
if ($ImmediateDirection -and $InputStride -eq 2) { $argsDocker += '--immediate-direction' }
if ($AsyncDirection -and $ImmediateDirection -and $InputStride -eq 2) { $argsDocker += '--async-direction' }
& docker @argsDocker
if ($LASTEXITCODE -ne 0) { throw 'COPY_MAIN failed; inspect result logs' }
