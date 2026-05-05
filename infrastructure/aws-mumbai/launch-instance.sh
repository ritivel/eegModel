#!/usr/bin/env bash
# launch-instance.sh — boot a p5.48xlarge into the capacity reservation.
#
# Once .env has CB_RESERVATION_ID set, this script can run any time during
# the 168 h window. The instance comes up, runs user-data (which trickles
# repo + bootstrap.sh on first boot), and you ssh in once it's reachable.
#
# Usage:
#   source .env
#   bash launch-instance.sh                # uses defaults from .env
#   DRY_RUN=1 bash launch-instance.sh      # show the command, don't run
#
# Re-running this will create a SECOND instance — the capacity reservation
# is for 1 instance only, so the second will fail with capacity-not-found.
# That's fine: monitor.sh prints the existing one if it's already up.

set -euo pipefail
cd "$(dirname "$0")"

for var in CB_RESERVATION_ID CB_INSTANCE_TYPE EC2_AMI_ID EC2_KEY_NAME EC2_SG_ID CB_SUBNET_ID; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: $var is empty in .env (purchase-capacity-block.sh fills CB_RESERVATION_ID)"
    exit 1
  fi
done

# Generate user-data: a tiny launcher that pulls bootstrap.sh from the repo
# and runs it as the ubuntu user. We keep user-data minimal (cloud-init has
# size and reliability quirks); the real work is in bootstrap.sh which
# lives in git.

USER_DATA=$(cat <<EOF
#!/usr/bin/env bash
set -euxo pipefail
exec > /var/log/eeg-bootstrap.log 2>&1
echo "[user-data] starting at \$(date -u)"
mkdir -p /opt/eeg-config
cat > /opt/eeg-config/env <<INNEREOF
export AWS_REGION="$AWS_REGION"
export S3_MUMBAI_BUCKET="$S3_MUMBAI_BUCKET"
export S3_MUMBAI_REGION="$S3_MUMBAI_REGION"
export S3_WAREHOUSE_BUCKET="$S3_WAREHOUSE_BUCKET"
export S3_WAREHOUSE_REGION="$S3_WAREHOUSE_REGION"
export EXP03_S3_BUCKET="$EXP03_S3_BUCKET"
export EXP03_S3_REGION="$EXP03_S3_REGION"
export EXP03_DATA_ROOT="$REMOTE_DATA_ROOT"
export REMOTE_REPO_URL="$REMOTE_REPO_URL"
export REMOTE_REPO_BRANCH="$REMOTE_REPO_BRANCH"
${WANDB_API_KEY:+export WANDB_API_KEY="$WANDB_API_KEY"}
${HF_TOKEN:+export HF_TOKEN="$HF_TOKEN"}
INNEREOF
chmod 600 /opt/eeg-config/env
chown ubuntu:ubuntu /opt/eeg-config/env

# Drop a marker so we know the user-data ran.
touch /opt/eeg-config/user-data-ok
echo "[user-data] env written; bootstrap.sh runs after first SSH (idempotent)."
EOF
)

# Tag spec — at minimum a Name tag so it's findable in the console + monitor.sh.
TAGS="ResourceType=instance,Tags=[{Key=Name,Value=$EC2_NAME_TAG},{Key=cluster,Value=eeg-mumbai-2026w19},{Key=billing,Value=research}]"

# Block device mapping: the DLAMI default is fine, but we bump root to 200 GB
# so `pip install` + HF cache + repo work without surprises. NVMe is separate
# and 30 TB by default on p5.48xlarge, no override needed.
BDM='[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true,"Iops":3000,"Throughput":250}}]'

CMD=(aws ec2 run-instances
  --region "$AWS_REGION"
  --image-id "$EC2_AMI_ID"
  --instance-type "$CB_INSTANCE_TYPE"
  --key-name "$EC2_KEY_NAME"
  --security-group-ids "$EC2_SG_ID"
  --subnet-id "$CB_SUBNET_ID"
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CB_RESERVATION_ID}"
  --block-device-mappings "$BDM"
  --tag-specifications "$TAGS"
  --user-data "$USER_DATA"
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled"
  --output json)

if [[ -n "${EC2_INSTANCE_PROFILE:-}" ]]; then
  CMD+=(--iam-instance-profile "Name=$EC2_INSTANCE_PROFILE")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY RUN — would execute:"
  printf '  %q ' "${CMD[@]}"; echo
  exit 0
fi

echo "[launch] starting $CB_INSTANCE_TYPE in $CB_AZ via reservation $CB_RESERVATION_ID..."
RESP=$("${CMD[@]}")
INSTANCE_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")

echo "[launch] InstanceId = $INSTANCE_ID"
echo "[launch] waiting for 'running' state..."
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"

PUB=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "[launch] public IP: $PUB"

# Wait for sshd; user-data writes /opt/eeg-config/user-data-ok before being
# done. We also need the ubuntu account to be ready. ~30-60 s typical.
echo "[launch] waiting for SSH..."
for i in $(seq 1 30); do
  if ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=5 -o BatchMode=yes ubuntu@"$PUB" "true" 2>/dev/null; then
    echo "[launch] SSH up after ${i}0 s"
    break
  fi
  sleep 10
done

cat <<EOF

==========================================
  Instance is up.

  ssh -A -i $EC2_KEY_PATH ubuntu@$PUB

  On the box, finish setup:
      git clone $REMOTE_REPO_URL ~/eegModel
      cd ~/eegModel
      git checkout $REMOTE_REPO_BRANCH
      bash infrastructure/aws-mumbai/bootstrap.sh

  Or (faster if your repo's already on github) one-shot from local Mac:
      ssh -A -i $EC2_KEY_PATH ubuntu@$PUB \\
        'git clone $REMOTE_REPO_URL ~/eegModel && cd ~/eegModel && \\
         git checkout $REMOTE_REPO_BRANCH && bash infrastructure/aws-mumbai/bootstrap.sh'

  Persisted state on the box:
      $REMOTE_DATA_ROOT     (NVMe, 30 TB, ephemeral — wiped on stop)
      ~/.aws               (you'll want to mount/copy your creds; or use --instance-profile)

  Public IP for monitoring scripts is also written to ~/.ssh/eeg-mumbai-host
EOF
echo "$PUB" > ~/.ssh/eeg-mumbai-host
echo "$INSTANCE_ID" > ~/.ssh/eeg-mumbai-instance-id
