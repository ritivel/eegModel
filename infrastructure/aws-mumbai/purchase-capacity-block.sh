#!/usr/bin/env bash
# purchase-capacity-block.sh — re-verify the offering, then commit the buy.
#
# This is the ONE irreversible script in this directory. After it returns
# success AWS bills $CB_UPFRONT_FEE_USD upfront and that money is gone. Run
# only when you're ready to commit.
#
# Usage:
#   source .env
#   bash purchase-capacity-block.sh             # interactive confirm
#   bash purchase-capacity-block.sh --yes       # skip confirm (CI / scripted)
#
# After success, this script writes the resulting CapacityReservationId
# back into .env (replacing the empty CB_RESERVATION_ID line) so
# launch-instance.sh can pick it up.

set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${CB_OFFERING_ID:-}" ]]; then
  echo "ERROR: source .env first (CB_OFFERING_ID not set)"; exit 1
fi

YES=0
[[ "${1:-}" == "--yes" ]] && YES=1

echo "==========================================="
echo "  AWS Capacity Block — Purchase"
echo "==========================================="
echo "  region:        $AWS_REGION"
echo "  offering:      $CB_OFFERING_ID"
echo "  instance:      $CB_INSTANCE_TYPE  (1× ${CB_INSTANCE_TYPE} = 8 H100)"
echo "  AZ:            $CB_AZ"
echo "  duration:      $CB_DURATION_HOURS hours"
echo "  start window:  $CB_START_DATE"
echo "  end window:    $CB_END_DATE"
echo "  upfront fee:   \$${CB_UPFRONT_FEE_USD} USD  (NON-REFUNDABLE)"
echo "==========================================="

# Re-verify the offering still exists at the same price. If AWS has
# updated prices (every July) or someone else bought this exact slot, we
# bail before spending.
echo "[1/3] re-verifying offering is still on the menu..."
LIVE=$(aws ec2 describe-capacity-block-offerings \
  --instance-type "$CB_INSTANCE_TYPE" \
  --instance-count 1 \
  --capacity-duration-hours "$CB_DURATION_HOURS" \
  --start-date-range "$(date -u -v -1H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%SZ)" \
  --end-date-range  "$(date -u -v +60d +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '+60 days' +%Y-%m-%dT%H:%M:%SZ)" \
  --region "$AWS_REGION" \
  --query "CapacityBlockOfferings[?CapacityBlockOfferingId=='$CB_OFFERING_ID']" \
  --output json)

if [[ "$LIVE" == "[]" ]] || [[ -z "$LIVE" ]]; then
  echo "ERROR: offering $CB_OFFERING_ID is no longer in the catalog."
  echo "       Re-run scripts/describe-offerings.sh to find a fresh one."
  exit 2
fi

LIVE_FEE=$(echo "$LIVE" | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['UpfrontFee'])")
if [[ "$LIVE_FEE" != "$CB_UPFRONT_FEE_USD" ]]; then
  echo "WARN: price drifted: env says \$$CB_UPFRONT_FEE_USD, AWS says \$$LIVE_FEE."
  echo "      Aborting. Update .env if the new price is acceptable, then retry."
  exit 3
fi
echo "      OK — \$$LIVE_FEE matches .env"

# Hard-confirm gate.
if [[ $YES -ne 1 ]]; then
  read -rp "[2/3] Type the literal word PURCHASE to commit: " CONFIRM
  if [[ "$CONFIRM" != "PURCHASE" ]]; then
    echo "      aborted (you typed: '$CONFIRM')"; exit 4
  fi
fi

echo "[3/3] purchasing..."
RESP=$(aws ec2 purchase-capacity-block \
  --capacity-block-offering-id "$CB_OFFERING_ID" \
  --instance-platform "Linux/UNIX" \
  --region "$AWS_REGION" \
  --output json)
echo "$RESP" | python3 -m json.tool

CR_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['CapacityReservation']['CapacityReservationId'])")
if [[ -z "$CR_ID" ]] || [[ "$CR_ID" == "None" ]]; then
  echo "ERROR: didn't get a CapacityReservationId back. Check AWS console."
  exit 5
fi

echo
echo "  ✓ purchased. CapacityReservationId = $CR_ID"

# Write CR_ID back into .env so launch-instance.sh can use it.
if [[ -f .env ]]; then
  if grep -q "^export CB_RESERVATION_ID=" .env; then
    # Use a portable in-place replace (gnu/bsd sed differ).
    python3 -c "
import re, pathlib
p = pathlib.Path('.env')
p.write_text(re.sub(r'^export CB_RESERVATION_ID=.*\$',
                    f'export CB_RESERVATION_ID=\"$CR_ID\"',
                    p.read_text(), flags=re.M))
"
    echo "  ✓ .env updated: CB_RESERVATION_ID=$CR_ID"
  else
    echo "export CB_RESERVATION_ID=\"$CR_ID\"" >> .env
    echo "  ✓ .env appended: CB_RESERVATION_ID=$CR_ID"
  fi
fi

echo
echo "Next:  launch-instance.sh (any time on or after $CB_START_DATE)"
