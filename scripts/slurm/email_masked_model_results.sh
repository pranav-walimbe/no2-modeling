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
result_plot="${run_dir}/results.png"

if [[ ! -s "${result_plot}" ]]; then
    echo "Expected result artifact is missing or empty: ${result_plot}" >&2
    exit 1
fi

echo "Emailing masked-model results to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'Masked NO2 model training completed successfully.\n\nRun: %s\nJob: %s\n\nThe attached figure summarizes training and compares masked-pixel errors with bilinear interpolation.\n' \
    "${run_dir}" \
    "${job_id}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'Masked NO2 model results (${job_id})' \
            -a '${result_plot}' '${recipient}'"

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
echo "Confirmed SMTP delivery of masked-model results to ${recipient}"
