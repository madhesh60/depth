#!/usr/bin/env bash
# secrets.sh — read one of the demo's secrets from SSM Parameter Store (SecureString; deploy.sh creates them).
# The value goes to your terminal only; nothing is written to disk.
#
#   ./infra/secrets.sh pin      # the approver PIN - put it in the submission form for the judges
#   ./infra/secrets.sh mcp      # the MCP bearer token for remote agents
#   ./infra/secrets.sh admin    # the admin token (webhooks, the load test)
set -euo pipefail
cd "$(dirname "$0")/.."
. infra/lib.sh

STACK="${STACK:-depth}"
case "${1:-}" in
  pin)   NAME=approver-pin ;;
  mcp)   NAME=mcp-token ;;
  admin) NAME=admin-token ;;
  *)     die "usage: ./infra/secrets.sh pin|mcp|admin" ;;
esac
awsq ssm get-parameter --name "/depth/$STACK/$NAME" --with-decryption --query Parameter.Value --output text \
  || die "no /depth/$STACK/$NAME yet - APPLY=1 ./infra/deploy.sh creates it"
