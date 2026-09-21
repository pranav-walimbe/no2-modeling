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
tabular_loss_plot="${run_dir}/tabular_loss_curve.png"
random_loss_plot="${run_dir}/random_init_delta_loss_curve.png"
pretrained_loss_plot="${run_dir}/pretrained_encoder_delta_loss_curve.png"
confusion_plot="${run_dir}/classification_confusion.png"

for artifact in \
    "${comparison_plot}" \
    "${tabular_loss_plot}" \
    "${random_loss_plot}" \
    "${pretrained_loss_plot}" \
    "${confusion_plot}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected result artifact is missing or empty: ${artifact}" >&2
        exit 1
    fi
done

echo "Emailing three-model classification results to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'NO2 classification training completed successfully.\n\nThe attachments compare the tabular MLP, random-init fusion model, and pretrained-encoder fusion model.\n\nRun: %s\nJob: %s\n' \
    "${run_dir}" \
    "${job_id}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 classification results (${job_id})' \
            -a '${comparison_plot}' -a '${tabular_loss_plot}' \
            -a '${random_loss_plot}' -a '${pretrained_loss_plot}' \
            -a '${confusion_plot}' '${recipient}'"

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
