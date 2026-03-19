#!/bin/bash
# Refresh the GCP identity token in .env and restart workers to pick it up.
# Run via cron every 45 minutes:
#   */45 * * * * /root/motivated-seller-app/backend/scripts/refresh_gcp_token.sh

set -e
ENV_FILE="/root/motivated-seller-app/.env"
TOKEN=$(gcloud auth print-identity-token 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "Failed to get token" >&2
    exit 1
fi

# Update token in .env
sed -i "s|^GCP_PROXY_TOKEN=.*|GCP_PROXY_TOKEN=$TOKEN|" "$ENV_FILE"

# Restart worker to pick up new token
cd /root/motivated-seller-app
docker compose restart worker 2>/dev/null

echo "Token refreshed ($(date))"
