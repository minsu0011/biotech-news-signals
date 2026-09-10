[CmdletBinding()]
param(
    [int]$InitialBackoffSeconds = 30,
    [int]$MaximumBackoffSeconds = 900
)

$ErrorActionPreference = 'Stop'
$Workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$Controller = Join-Path $Workspace 'autonomous_v37plus.py'
$FinalMarker = Join-Path $Workspace 'state\FINAL_PASS'
$Backoff = $InitialBackoffSeconds

Set-Location -LiteralPath $Workspace
while ($true) {
    & python $Controller --resume --start-version 37 --continue-until-pass
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0 -and (Test-Path -LiteralPath $FinalMarker)) {
        & python -c "import autonomous_v37plus as a,sys;sys.exit(0 if a.verify_final_marker() else 1)"
        if ($LASTEXITCODE -eq 0) { break }
    }

    if ($ExitCode -eq 130) { break }
    if ($ExitCode -eq 20) {
        # No version is consumed while a material runner/data intervention is absent.
        $Backoff = [Math]::Min($MaximumBackoffSeconds, [Math]::Max(300, $Backoff * 2))
    } else {
        $Backoff = [Math]::Min($MaximumBackoffSeconds, [Math]::Max($InitialBackoffSeconds, $Backoff * 2))
    }
    Start-Sleep -Seconds $Backoff
}
