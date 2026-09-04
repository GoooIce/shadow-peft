#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
remote_host=${MAGE_ES_PEER_HOST:-goooinext}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

while true; do
  complete=true
  for episode in $(seq 0 11); do
    if [[ ! -f "$output_root/gradient-episode-$episode-validation.json" ]]; then
      complete=false
      break
    fi
  done
  if [[ "$complete" == true ]]; then
    for episode in $(seq 12 23); do
      if ! ssh "$remote_host" test -f \
        "$output_root/gradient-episode-$episode-validation.json"; then
        complete=false
        break
      fi
    done
  fi
  if [[ "$complete" == true ]]; then
    break
  fi
  sleep 60
done

for episode in $(seq 12 23); do
  mkdir -p "$output_root/gradient-episode-$episode"
  rsync -a --partial \
    "$remote_host:$output_root/gradient-episode-$episode/" \
    "$output_root/gradient-episode-$episode/"
  rsync -a \
    "$remote_host:$output_root/gradient-episode-$episode-validation.json" \
    "$output_root/gradient-episode-$episode-validation.json"

  remote_manifest=$(
    ssh "$remote_host" \
      "cd '$output_root' && find 'gradient-episode-$episode' -type f -print0 | sort -z | xargs -0 sha256sum"
  )
  local_manifest=$(
    cd "$output_root"
    find "gradient-episode-$episode" -type f -print0 | sort -z | xargs -0 sha256sum
  )
  if [[ "$local_manifest" != "$remote_manifest" ]]; then
    echo "source artifact hash mismatch after peer sync: episode $episode" >&2
    exit 1
  fi
done

"$repo_root/scripts/launch_mage_flow_es_20x_finalize_gb10.sh" "$output_root"
