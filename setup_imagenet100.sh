#!/bin/bash
set -e

# Compatibility entry point; keep dataset preparation in scripts/.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/scripts/setup_imagenet100.sh"
