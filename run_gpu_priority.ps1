$ErrorActionPreference = "Stop"

$pythonPath = "$PSScriptRoot\venv\Scripts\python.exe"
$runPath = "$PSScriptRoot\run.py"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$chunkSize = 25000

# Alternar os tipos impede que textos únicos e longos de uma origem atrasem
# os maiores grupos duplicados e os textos curtos da outra origem.
for ($cycle = 1; $cycle -le 8; $cycle++) {
    & $pythonPath `
        $runPath run `
        --content posts `
        --platform whatsapp `
        --platform telegram `
        --limit $chunkSize `
        --run-name "producao_lote16_posts_${runStamp}_${cycle}" `
        --skip-migrate

    $postsExitCode = $LASTEXITCODE
    if ($postsExitCode -notin @(0, 2)) { exit $postsExitCode }

    & $pythonPath `
        $runPath run `
        --content comments `
        --limit $chunkSize `
        --run-name "producao_lote16_comments_${runStamp}_${cycle}" `
        --skip-migrate

    $commentsExitCode = $LASTEXITCODE
    if ($commentsExitCode -notin @(0, 2)) { exit $commentsExitCode }
}

# Captura posts de outras plataformas e qualquer erro isolado ainda pendente.
& $pythonPath `
    $runPath run `
    --content all `
    --run-name "producao_lote16_final_${runStamp}" `
    --skip-migrate

exit $LASTEXITCODE
