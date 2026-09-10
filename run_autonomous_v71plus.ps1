[CmdletBinding()]
param(
    [double]$MinimumUsefulHours = 10.0,
    [int]$InitialBackoffSeconds = 30,
    [int]$MaximumBackoffSeconds = 900
)

$ErrorActionPreference = 'Stop'
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Controller = Join-Path $Workspace 'autonomous_v71plus.py'
$FinalMarker = Join-Path $Workspace 'state\FINAL_PASS'
$Backoff = $InitialBackoffSeconds

Set-Location -LiteralPath $Workspace
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
    if ([string]::IsNullOrWhiteSpace(
            [Environment]::GetEnvironmentVariable($Name, 'Process'))) {
        $UserValue = [Environment]::GetEnvironmentVariable($Name, 'User')
        if (-not [string]::IsNullOrWhiteSpace($UserValue)) {
            [Environment]::SetEnvironmentVariable($Name, $UserValue, 'Process')
        }
    }
}
# Alpaca credentials may have been rotated while this watchdog was stopped.
# User scope is authoritative for these four names; overwrite inherited stale
# process values without ever printing them.
$AlpacaEnvironmentNames = @(
    'ALPACA_API_KEY',
    'ALPACA_API_SECRET',
    'APCA_API_KEY_ID',
    'APCA_API_SECRET_KEY'
)
foreach ($Name in $AlpacaEnvironmentNames) {
    $UserValue = [Environment]::GetEnvironmentVariable($Name, 'User')
    if (-not [string]::IsNullOrWhiteSpace($UserValue)) {
        [Environment]::SetEnvironmentVariable($Name, $UserValue, 'Process')
    } else {
        [Environment]::SetEnvironmentVariable($Name, $null, 'Process')
    }
}
$MissingProviderEnvironment = @($ProviderEnvironmentNames | Where-Object {
    [string]::IsNullOrWhiteSpace(
        [Environment]::GetEnvironmentVariable($_, 'Process'))
})
if ($MissingProviderEnvironment.Count -ne 0) {
    throw 'Provider environment propagation gate failed; values were not logged.'
}
while ($true) {
    & python $Controller --resume --continue-until-pass --min-useful-hours $MinimumUsefulHours
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0 -and (Test-Path -LiteralPath $FinalMarker)) {
        & python -c "import autonomous_v37plus as a,sys;sys.exit(0 if a.verify_final_marker() else 1)"
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($ExitCode -eq 130) { break }

    $Backoff = [Math]::Min($MaximumBackoffSeconds,
        [Math]::Max($InitialBackoffSeconds, $Backoff * 2))
    Start-Sleep -Seconds $Backoff
}
