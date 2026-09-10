param(
    [int]$MaxRequests = 200,
    [int]$MaxLocalItems = 2500,
    [int]$MaxDiscoveryQueries = 50,
    [int]$Workers = 32,
    [int]$PreviewSeconds = 1800,
    [int]$EmbeddingMaxItems = 5000,
    [int]$EmbeddingBatchSize = 64,
    [string]$EmbeddingDevice = 'auto',
    [int]$IdleSeconds = 300,
    [int]$RestartBackoffSeconds = 60
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = 'C:\Users\minsu\anaconda3\python.exe'
$Runner = Join-Path $ProjectRoot 'historical_news_10h_orchestrator.py'

Set-Location -LiteralPath $ProjectRoot

while ($true) {
    & $PythonExe -B $Runner --resume --target-productive-seconds 36000 --checkpoint-seconds 7200 --max-requests $MaxRequests --max-local-items $MaxLocalItems --max-discovery-queries $MaxDiscoveryQueries --workers $Workers --preview-seconds $PreviewSeconds --embedding-max-items $EmbeddingMaxItems --embedding-batch-size $EmbeddingBatchSize --embedding-device $EmbeddingDevice --idle-seconds $IdleSeconds
    $RunnerExitCode = $LASTEXITCODE

    if ($RunnerExitCode -in @(0, 10, 12)) {
        exit $RunnerExitCode
    }

    Start-Sleep -Seconds $RestartBackoffSeconds
}
