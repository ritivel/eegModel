#!/usr/bin/env bash
# monitor.sh — quick local-Mac status dashboard for the reservation.
#
# Shows: instance state, time elapsed, time remaining, $/h burn (paid
# upfront so this is a "value left on the meter" indicator, not a bill).
#
# Usage:
#   source .env
#   bash monitor.sh
#   watch -n 60 'cd ~/Downloads/Code/eegModel/infrastructure/aws-mumbai && bash monitor.sh'

set -uo pipefail
cd "$(dirname "$0")"

[[ -z "${CB_RESERVATION_ID:-}" ]] && { echo "ERROR: CB_RESERVATION_ID empty in .env"; exit 1; }

NOW=$(date -u +%s)
START=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$CB_START_DATE" +%s 2>/dev/null \
        || date -u -d "$CB_START_DATE" +%s)
END=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$CB_END_DATE" +%s 2>/dev/null \
        || date -u -d "$CB_END_DATE" +%s)

H_ELAPSED=$(( (NOW - START) / 3600 ))
H_REMAINING=$(( (END - NOW) / 3600 ))
COST_PER_H=$(python3 -c "print(round($CB_UPFRONT_FEE_USD/$CB_DURATION_HOURS, 2))")
COST_REMAINING=$(python3 -c "print(round($COST_PER_H*max($H_REMAINING,0), 2))")

echo "==========================================="
echo "  Reservation: $CB_RESERVATION_ID"
echo "  Window:      $CB_START_DATE  →  $CB_END_DATE"
echo "  Elapsed:     ${H_ELAPSED} h   |   Remaining: ${H_REMAINING} h"
echo "  \$/hr equivalent: \$${COST_PER_H}    Value left on meter: \$${COST_REMAINING}"
echo "==========================================="

# Reservation state
echo
echo "-- capacity reservation --"
aws ec2 describe-capacity-reservations \
  --capacity-reservation-ids "$CB_RESERVATION_ID" \
  --region "$AWS_REGION" \
  --query "CapacityReservations[0].{State:State, AvailableInstances:AvailableInstanceCount, TotalInstances:TotalInstanceCount, EndDate:EndDate}" \
  --output table 2>/dev/null

# Live instance(s)
echo
echo "-- instance(s) --"
aws ec2 describe-instances \
  --filters "Name=tag:cluster,Values=eeg-mumbai-2026w19" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --region "$AWS_REGION" \
  --query "Reservations[].Instances[].{Id:InstanceId, State:State.Name, IP:PublicIpAddress, Up:LaunchTime, Type:InstanceType}" \
  --output table 2>/dev/null

# Mumbai S3 mirror size (skip if no creds)
echo
echo "-- Mumbai mirror size --"
aws s3 ls "s3://${S3_MUMBAI_BUCKET}/derived/" --recursive --summarize \
  --region "$S3_MUMBAI_REGION" 2>/dev/null | tail -2

if [[ -f ~/.ssh/eeg-mumbai-host ]]; then
  PUB=$(cat ~/.ssh/eeg-mumbai-host)
  echo
  echo "-- SSH --"
  echo "  ssh -A -i $EC2_KEY_PATH ubuntu@$PUB"
fi

if [[ $H_REMAINING -le 1 && $H_REMAINING -gt -2 ]]; then
  echo
  echo "  *** WARNING: less than 1 hour left. Run teardown.sh NOW. ***"
fi
