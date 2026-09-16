#!/bin/bash

set -euo pipefail

artifact="/global/home/users/pranavwalimbe/vis/model-image-value-38739847/dashboard.png"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

if [[ ! -s "${artifact}" ]]; then
    echo "Dashboard is missing or empty: ${artifact}" >&2
    exit 1
fi

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'The CNN versus XGBoost raster-information dashboard is attached.\n\nAnalysis job: 38739847\n' \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 model image-value dashboard' -a '${artifact}' '${recipient}'"

for _ in {1..30}; do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        tail -c "+$((mail_log_offset + 1))" "${mail_log}" \
        | grep -F "to=<${recipient}>" \
        | grep -Fq 'status=sent'; then
        echo "Confirmed SMTP delivery of dashboard to ${recipient}"
        exit 0
    fi
    sleep 1
done

echo "Could not confirm dashboard delivery in ${mail_log}" >&2
exit 1
