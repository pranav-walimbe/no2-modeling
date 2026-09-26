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
accuracy_plot="${run_dir}/split_class_accuracy.png"
loss_plot="${run_dir}/training_curves.png"
characteristic_plot="${run_dir}/accuracy_by_characteristic.png"

for artifact in \
    "${accuracy_plot}" \
    "${loss_plot}" \
    "${characteristic_plot}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected result artifact is missing or empty: ${artifact}" >&2
        exit 1
    fi
done

echo "Emailing seasonal and vision-seasonal results to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'NO2 classification training completed successfully.\n\nThe attachments compare seasonal and vision-seasonal accuracy, training loss, AOI characteristics, and monthly validation and test accuracy.\n\nRun: %s\nJob: %s\n' \
    "${run_dir}" \
    "${job_id}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 classification results (${job_id})' \
            -a '${accuracy_plot}' -a '${loss_plot}' -a '${characteristic_plot}' '${recipient}'"

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
