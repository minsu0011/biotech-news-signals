[CmdletBinding()]
param(
    [int]$InitialBackoffSeconds = 30,
    [int]$MaximumBackoffSeconds = 900
)

$ErrorActionPreference = 'Stop'
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Controller = Join-Path $Workspace 'run_v224_dual_track_8hplus.py'
$Backoff = $InitialBackoffSeconds
$PythonCandidates = @(
    (Join-Path ([Environment]::GetFolderPath('UserProfile')) 'anaconda3\python.exe'),
    (Join-Path ([Environment]::GetFolderPath('UserProfile')) 'miniconda3\python.exe')
)
$Python = $PythonCandidates | Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($Python)) {
    throw 'A real Python interpreter was not found (WindowsApps aliases are rejected).'
}

Set-Location -LiteralPath $Workspace

# Import only already-provisioned user-scope credentials. Values are never
# rendered, persisted by this watchdog, or passed on a command line.
$ProviderEnvironmentNames = @(
    'MASSIVE_API_KEY',
    'POLYGON_API_KEY',
    'KIS_APP_KEY',
    'KIS_APP_SECRET',
    'ALPACA_API_KEY',
    'ALPACA_API_SECRET',
    'APCA_API_KEY_ID',
    'APCA_API_SECRET_KEY'
)
foreach ($Name in $ProviderEnvironmentNames) {
    $UserValue = [Environment]::GetEnvironmentVariable($Name, 'User')
    if (-not [string]::IsNullOrWhiteSpace($UserValue)) {
        [Environment]::SetEnvironmentVariable($Name, $UserValue, 'Process')
    }
}

while ($true) {
    & $Python $Controller --resume
    $ExitCode = $LASTEXITCODE

    # 0: minimum productive runtime completed; 10: frozen-integrity fail-closed;
    # 12: another scheduler owns the lock; 130: explicit interruption.
    # None should be restarted blindly.
    if ($ExitCode -in @(0, 10, 12, 130)) {
        break
    }

    Start-Sleep -Seconds $Backoff
    $Backoff = [Math]::Min(
        $MaximumBackoffSeconds,
        [Math]::Max($InitialBackoffSeconds, $Backoff * 2)
    )
}
