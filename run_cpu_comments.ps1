$ErrorActionPreference = "Stop"

$env:OLLAMA_URL = "http://127.0.0.1:11435/api/generate"
$env:OLLAMA_NUM_GPU = "0"
$env:OLLAMA_NUM_CTX = "2304"

& "$PSScriptRoot\venv\Scripts\python.exe" `
    "$PSScriptRoot\run.py" run `
    --content comments `
    --run-name producao_cpu_comments_20260722 `
    --skip-migrate

exit $LASTEXITCODE
