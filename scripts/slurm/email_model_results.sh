#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 RUN_DIR JOB_ID RECIPIENT" >&2
    exit 2
fi

run_dir="$1"
job_id="$2"
recipient="$3"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
comparison_plot="${run_dir}/model_comparison.png"
prediction_plot="${run_dir}/regression_predictions.png"
raster_loss_plot="${run_dir}/loss_curve.png"
tabular_loss_plot="${run_dir}/tabular_loss_curve.png"
hurdle_diagnostics_plot="${run_dir}/hurdle_diagnostics.png"

for artifact in \
    "${comparison_plot}" \
    "${prediction_plot}" \
    "${raster_loss_plot}" \
    "${tabular_loss_plot}" \
    "${hurdle_diagnostics_plot}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected result artifact is missing or empty: ${artifact}" >&2
        exit 1
    fi
done

echo "Emailing regression plots and both loss curves to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'NO2 hurdle training completed successfully.\n\nThe attachments compare the raster hurdle model with the LDS-weighted MLP and report gate and conditional-magnitude diagnostics.\n\nRun: %s\nJob: %s\n' \
    "${run_dir}" \
    "${job_id}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 regression results (${job_id})' \
            -a '${comparison_plot}' -a '${prediction_plot}' -a '${raster_loss_plot}' \
            -a '${tabular_loss_plot}' -a '${hurdle_diagnostics_plot}' '${recipient}'"

delivery_confirmed=false
for _ in {1..30}; do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        tail -c "+$((mail_log_offset + 1))" "${mail_log}" \
        | grep -F "to=<${recipient}>" \
        | grep -Fq 'status=sent'; then
        delivery_confirmed=true
        break
    fi
    sleep 1
done
if [[ "${delivery_confirmed}" != true ]]; then
    echo "Could not confirm successful SMTP delivery in ${mail_log}" >&2
    exit 1
fi
echo "Confirmed SMTP delivery of model results to ${recipient}"
