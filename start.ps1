$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$ragPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython)) {
    Write-Host '请先运行：python -m venv .venv'
    Write-Host '然后运行：.\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 1
}
& $ragPython -m streamlit run app.py
exit $LASTEXITCODE
