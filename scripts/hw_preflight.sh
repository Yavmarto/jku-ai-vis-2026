#!/usr/bin/env bash
# Checks that the AKD1000 card is usable before the chip tests run.
# Usage: scripts/hw_preflight.sh [python-binary]   (default: python3)
set -euo pipefail

PY="${1:-python3}"

echo "== PCI device (AKD1000 is 1e7c:bca1) =="
if ! lspci -nn -d 1e7c:bca1 | grep -q .; then
  echo "AKD1000 not found on the PCI bus" >&2
  exit 1
fi
lspci -nn -d 1e7c:bca1

echo "== Kernel module =="
# The module name is unverified; this only prints what is loaded.
lsmod | grep -i akida || echo "no module with 'akida' in its name is listed"

echo "== akida package sees the device =="
"$PY" - <<'EOF'
import akida
devices = akida.devices()
print("akida", getattr(akida, "__version__", "unknown"), "devices:", devices)
raise SystemExit(0 if devices else "no Akida device visible to the akida package")
EOF
