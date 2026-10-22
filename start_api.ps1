param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [ValidateNotNullOrEmpty()]
    [string]$ListenAddress = '127.0.0.1'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$ragPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython)) {
    Write-Host '请先创建虚拟环境并安装 requirements.txt 中的依赖。'
    exit 1
}
Write-Host "API 检查地址：http://${ListenAddress}:$Port/health"
Write-Host '监听局域网地址时，其他设备还需要能够通过电脑防火墙访问此端口。'
& $ragPython -m uvicorn api:app --host $ListenAddress --port $Port --workers 1
exit $LASTEXITCODE
