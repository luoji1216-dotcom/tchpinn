param(
    [string]$Pattern = "results_5_3_3_austria_1GHz_direct_*"
)

$ErrorActionPreference = "Stop"

Get-ChildItem -Path $PSScriptRoot -Directory -Filter $Pattern |
    ForEach-Object {
        $history = Join-Path $_.FullName "loss_history.csv"
        if (-not (Test-Path $history)) {
            return
        }
        Import-Csv $history | ForEach-Object {
            [pscustomobject]@{
                Run = Split-Path -Leaf (Split-Path -Parent $history)
                Step = [int][double]$_.step
                RelError = [double]$_.rel_error_continuous
                SSIM = [double]$_.ssim_continuous
                RelErrorThresholded = [double]$_.rel_error_thresholded
                SSIMThresholded = [double]$_.ssim_thresholded
            }
        }
    } |
    Sort-Object RelError, @{ Expression = "SSIM"; Descending = $true } |
    Select-Object -First 30 |
    ForEach-Object {
        [pscustomobject]@{
            Run = $_.Run
            Step = $_.Step
            RelError = "{0:F6}" -f $_.RelError
            SSIM = "{0:F6}" -f $_.SSIM
            RelErrorThresholded = "{0:F6}" -f $_.RelErrorThresholded
            SSIMThresholded = "{0:F6}" -f $_.SSIMThresholded
        }
    } |
    Format-Table -AutoSize -Wrap
