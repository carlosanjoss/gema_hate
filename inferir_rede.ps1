param(
    [Parameter(Position = 0)]
    [string]$Rede,

    [ValidateSet("all", "posts", "comments")]
    [string]$Conteudo = "all",

    [ValidateRange(1, 2147483647)]
    [int]$Limite,

    [switch]$SomenteContar,
    [switch]$SemConfirmacao
)

$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot
$pythonPath = Join-Path $projectDir "venv\Scripts\python.exe"
$runPath = Join-Path $projectDir "run.py"

$platformAliases = @{
    "reddit"    = "reddit"
    "reddi"     = "reddit"
    "twitter"   = "twitter"
    "x"         = "twitter"
    "youtube"   = "youtube"
    "facebook"  = "facebook"
    "tiktok"    = "tiktok"
    "tik tok"   = "tiktok"
    "whatsapp"  = "whatsapp"
    "whats"     = "whatsapp"
    "telegram"  = "telegram"
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error "Python do projeto nao encontrado em: $pythonPath"
    exit 1
}

if (-not $Rede) {
    Write-Host "Redes disponiveis: reddit, twitter, youtube, facebook, tiktok, whatsapp e telegram."
    $Rede = Read-Host "Digite a rede que deseja avaliar"
}

$normalizedInput = $Rede.Trim().ToLowerInvariant()
if (-not $platformAliases.ContainsKey($normalizedInput)) {
    Write-Host "Rede invalida: $Rede" -ForegroundColor Red
    Write-Host "Use: reddit, twitter, youtube, facebook, tiktok, whatsapp ou telegram."
    exit 2
}

$platform = $platformAliases[$normalizedInput]
$countArgs = @(
    $runPath,
    "run",
    "--content", $Conteudo,
    "--platform", $platform,
    "--dry-run"
)
if ($PSBoundParameters.ContainsKey("Limite")) {
    $countArgs += @("--limit", $Limite.ToString())
}

Write-Host ""
Write-Host "Verificando dados pendentes de $platform..." -ForegroundColor Cyan
& $pythonPath @countArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($SomenteContar) {
    exit 0
}

$activeWorkers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "^python(?:\.exe)?$" -and
    $_.CommandLine -match "run\.py\s+run" -and
    $_.CommandLine -notmatch "--dry-run"
}
if ($activeWorkers) {
    $processIds = ($activeWorkers.ProcessId | Sort-Object -Unique) -join ", "
    Write-Host ""
    Write-Host "Ja existe uma inferencia em execucao (PID: $processIds)." -ForegroundColor Yellow
    Write-Host "Aguarde ela terminar antes de iniciar outra para nao dividir a GPU."
    exit 3
}

if (-not $SemConfirmacao) {
    $answer = Read-Host "Iniciar agora somente para $platform? (S/N)"
    if ($answer.Trim().ToLowerInvariant() -notin @("s", "sim", "y", "yes")) {
        Write-Host "Execucao cancelada."
        exit 0
    }
}

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runName = "rede_${platform}_$runStamp"
$runArgs = @(
    $runPath,
    "run",
    "--content", $Conteudo,
    "--platform", $platform,
    "--run-name", $runName
)
if ($PSBoundParameters.ContainsKey("Limite")) {
    $runArgs += @("--limit", $Limite.ToString())
}

Write-Host ""
Write-Host "Iniciando inferencia de $platform..." -ForegroundColor Green
Write-Host "Nome da execucao: $runName"
Write-Host "Mantenha esta janela aberta. Pressione Ctrl+C somente se quiser interromper."
Write-Host ""

Push-Location $projectDir
try {
    & $pythonPath @runArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
