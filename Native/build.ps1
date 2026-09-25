param([string]$Architecture = 'native')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$outputDirectory = Join-Path $PSScriptRoot 'build'
New-Item -ItemType Directory -Force $outputDirectory | Out-Null
$nvccPath = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
if (-not $nvccPath -and $env:CUDA_PATH) { $nvccPath = Join-Path $env:CUDA_PATH 'bin\nvcc.exe' }
if (-not $nvccPath -or -not (Test-Path $nvccPath)) { throw 'Install the NVIDIA CUDA Toolkit and add nvcc to PATH.' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$compiler = if (Get-Command cl.exe -ErrorAction SilentlyContinue) { (Get-Command cl.exe).Source } elseif (Test-Path $vswhere) { & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe' | Select-Object -First 1 }
if (-not $compiler) { throw 'Install Visual Studio Desktop development with C++ and a Windows SDK.' }
$arguments = @('-std=c++17', '-O3', '-lineinfo', "-arch=$Architecture", '-ccbin', (Split-Path $compiler), '-Xcompiler', '/EHsc,/MT,/utf-8', (Join-Path $PSScriptRoot 'main.cu'), '-o', (Join-Path $outputDirectory 'ClearwaterNative.exe'), '-Xlinker', '/SUBSYSTEM:WINDOWS', '-lcudart_static', '-ld3d11', '-ldxgi', '-lwindowscodecs', '-lole32', '-luser32', '-lgdi32', '-lcomctl32')
& $nvccPath @arguments
if ($LASTEXITCODE -ne 0) { throw "Native compilation failed ($LASTEXITCODE)." }
New-Item -ItemType Directory -Force (Join-Path $outputDirectory 'assets') | Out-Null
Copy-Item (Join-Path $projectRoot 'assets\seabed.jpg') (Join-Path $outputDirectory 'assets\seabed.jpg')
Copy-Item (Join-Path $projectRoot 'LICENSE') $outputDirectory
Copy-Item (Join-Path $projectRoot 'THIRD_PARTY_NOTICES.md') $outputDirectory
Copy-Item (Join-Path $PSScriptRoot 'README.md') $outputDirectory
Write-Host "Built $outputDirectory\ClearwaterNative.exe"

