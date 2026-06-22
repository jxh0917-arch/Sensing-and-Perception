param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Target,

    [string]$Python = "python",

    [switch]$DryRun
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$argsList = @("run_pipeline.py", $Target, "--python", $Python)

if ($DryRun) {
    $argsList += "--dry-run"
}

Push-Location $repoRoot
try {
    & $Python @argsList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
