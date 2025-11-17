$OutDir = "experiments"
if (!(Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$ALPHAS = @(0.1, 0.05, 0.01)
$GAMMAS = @(0.95, 0.99)
$DECAYS = @(0.995, 0.99, 0.98)
$DISCRETS = @("A_coarse", "A_5000")

$EPISODES_A_coarse = 3000
$EPISODES_A_5000 = 5000

foreach ($discret in $DISCRETS) {
    foreach ($alpha in $ALPHAS) {
        foreach ($gamma in $GAMMAS) {
            foreach ($decay in $DECAYS) {

                $epis = if ($discret -eq "A_coarse") { $EPISODES_A_coarse } else { $EPISODES_A_5000 }

                $outname = "${discret}_alpha${alpha}_gamma${gamma}_decay${decay}_ep${epis}"
                Write-Host "Launching: $outname"

                $logPath = "$OutDir/$outname.log"

                poetry run python src/qlearning/train_qlearning.py `
                    --discret $discret `
                    --alpha $alpha `
                    --gamma $gamma `
                    --epsilon_decay $decay `
                    --episodes $epis `
                    --outdir $OutDir `
                    --seed 0 `
                    | Tee-Object -FilePath $logPath

                Write-Host "Finished: $outname"

                Start-Sleep -Seconds 2   # Evita problemas con gym
            }
        }
    }
}

Write-Host "All experiments finished."
Write-Host "Generating plots..."
poetry run python src/analysis/plot_results.py --indir experiments
Write-Host "Plots saved successfully."