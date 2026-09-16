#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 DASHBOARD JOB_ID RECIPIENT" >&2
    exit 2
fi

artifact="$1"
job_id="$2"
recipient="$3"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

if [[ ! -s "${artifact}" ]]; then
    echo "Flux dashboard is missing or empty: ${artifact}" >&2
    exit 1
fi

echo "Emailing flux evaluation dashboard to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'Current flux-model validation and test evaluation is attached.\n\nJob: %s\n' "${job_id}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 flux model evaluation (${job_id})' -a '${artifact}' '${recipient}'"

for _ in {1..30}; do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        tail -c "+$((mail_log_offset + 1))" "${mail_log}" \
        | grep -F "to=<${recipient}>" \
        | grep -Fq 'status=sent'; then
        echo "Confirmed SMTP delivery of flux dashboard to ${recipient}"
        exit 0
    fi
    sleep 1
done

echo "Could not confirm flux-dashboard delivery in ${mail_log}" >&2
exit 1
