#!/usr/bin/env bash

repo_root_from_script() {
  local script_path="$1"
  local dir
  dir="$(cd "$(dirname "$script_path")" && pwd)"
  if git -C "$dir" rev-parse --show-toplevel >/dev/null 2>&1; then
    git -C "$dir" rev-parse --show-toplevel
    return 0
  fi
  cd "$dir/../.." && pwd
}
