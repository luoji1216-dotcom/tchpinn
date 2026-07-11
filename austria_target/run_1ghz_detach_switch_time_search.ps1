param(
    [string]$Python = "T:\anaconda3\envs\py\python.exe",
    [string]$Device = "auto",
    [int]$Epochs = 6000,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$script = "austria_target\run_5_3_3_austria_1ghz_integral_direct.py"
$sourceDir = "austria_target\results_5_3_3_austria_1GHz_direct_route3_operator_A_detach_field_to60000"

$cases = @(
    @{ Label = "S50000"; Step = 50000 },
    @{ Label = "S51000"; Step = 51000 },
    @{ Label = "S52000"; Step = 52000 },
    @{ Label = "S53000"; Step = 53000 },
    @{ Label = "S54000"; Step = 54000 }
)

foreach ($case in $cases) {
    $checkpoint = Join-Path $sourceDir ("checkpoint_adam_{0:D6}.pt" -f $case.Step)
    $outputDir = "results_5_3_3_austria_1GHz_direct_switch_{0}_detach_lowint_to{1}" -f $case.Label, ($case.Step + $Epochs)

    $args = @(
        $script,
        "--device", $Device,
        "--epochs", "$Epochs",
        "--resume-checkpoint", $checkpoint,
        "--output-dir", $outputDir,
        "--lr", "1e-6",
        "--weight-data", "120",
        "--weight-pde", "0.006",
        "--weight-boundary", "0.02",
        "--weight-integral-data", "50",
        "--weight-tv", "0.001",
        "--weight-contrast-l1", "0.08",
        "--integral-internal-field-mode", "detach_field",
        "--log-every", "500",
        "--checkpoint-every", "500"
    )

    Write-Host ""
    Write-Host "=== $($case.Label): $checkpoint -> $outputDir ==="
    if ($DryRun) {
        Write-Host $Python $args
    } else {
        & $Python @args
    }
}
