#!/usr/bin/env bash
set -Eeuo pipefail

# =============================
# Config
# =============================
CUDA_VER="12.8.0"
CUDA_SHORT="12-8"
PIN_FILE="cuda-wsl-ubuntu.pin"
REPO_DEB="cuda-repo-wsl-ubuntu-${CUDA_SHORT}-local_${CUDA_VER}-1_amd64.deb"

PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/${PIN_FILE}"
DEB_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/${REPO_DEB}"

APT_PIN_DIR="/etc/apt/preferences.d"
APT_PIN_PATH="${APT_PIN_DIR}/cuda-repository-pin-600"
LOCAL_REPO_DIR="/var/cuda-repo-wsl-ubuntu-${CUDA_SHORT}-local"
KEYRING_DST="/usr/share/keyrings"
TOOLKIT_PKG="cuda-toolkit-${CUDA_SHORT}"

# =============================
# Helpers
# =============================
need_cmd() { command -v "$1" >/dev/null 2>&1; }
fetch() {
  local url="$1" out="$2"
  if need_cmd wget; then
    wget -O "$out" "$url"
  elif need_cmd curl; then
    curl -Lfo "$out" "$url"
  else
    echo "Error: need either wget or curl installed." >&2
    exit 1
  fi
}

require_sudo() {
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if need_cmd sudo; then
      sudo -v
    else
      echo "Error: this script needs root privileges (sudo not found)." >&2
      exit 1
    fi
  fi
}

remove_local_repo_sources() {
  echo "==> Removing any APT sources pointing to the local CUDA file repo..."
  local pattern="file:${LOCAL_REPO_DIR//\//\\/}"
  local tmp

  # 1) Remove matching .list files in sources.list.d
  sudo bash -c '
    set -Eeuo pipefail
    # Delete any .list explicitly referencing the local file repo
    matches=$(grep -RIl "^deb .*file:'"$LOCAL_REPO_DIR"'" /etc/apt/sources.list.d 2>/dev/null || true)
    [ -n "${matches:-}" ] && echo "$matches" | xargs -r rm -f
    # Common NVIDIA naming fallback
    rm -f /etc/apt/sources.list.d/cuda-*-local*.list 2>/dev/null || true
  '

  # 2) If /etc/apt/sources.list itself references the file repo, strip those lines
  if grep -qE "^deb .*${pattern}" /etc/apt/sources.list 2>/dev/null; then
    echo "==> Cleaning lines from /etc/apt/sources.list that point to ${LOCAL_REPO_DIR}..."
    tmp="$(mktemp)"
    sudo awk -v pat="${LOCAL_REPO_DIR}" '
      BEGIN { removed=0 }
      $0 ~ "^deb " && index($0, pat) { removed=1; next }
      { print }
      END {
        if (removed) {
          # notify via stderr
          # (awk cannot write to stderr portably; the echo below will handle logging)
        }
      }
    ' /etc/apt/sources.list > "${tmp}"
    sudo cp "${tmp}" /etc/apt/sources.list
    rm -f "${tmp}"
  fi
}

# =============================
# Temp workspace + trap
# =============================
WORKDIR="$(mktemp -d)"
cleanup_tmp() { rm -rf "$WORKDIR" || true; }
trap cleanup_tmp EXIT

# =============================
# Begin
# =============================
echo "==> Installing CUDA toolkit ${CUDA_SHORT} on WSL Ubuntu..."
require_sudo

# Pre-clean: remove any stale local repo references from previous runs
remove_local_repo_sources || true

# 1) Download and install APT pin
echo "==> Downloading NVIDIA APT pin..."
fetch "$PIN_URL" "${WORKDIR}/${PIN_FILE}"
echo "==> Placing APT pin at ${APT_PIN_PATH}..."
sudo mkdir -p "$APT_PIN_DIR"
sudo mv "${WORKDIR}/${PIN_FILE}" "$APT_PIN_PATH"

# 2) Download and install local repo .deb
echo "==> Downloading ${REPO_DEB}..."
fetch "$DEB_URL" "${WORKDIR}/${REPO_DEB}"

echo "==> Installing local repo package..."
sudo dpkg -i "${WORKDIR}/${REPO_DEB}" || {
  echo "dpkg failed; attempting to fix missing deps..." >&2
  sudo apt-get -y -f install
  sudo dpkg -i "${WORKDIR}/${REPO_DEB}"
}

# 3) Copy keyring
if [ -d "$LOCAL_REPO_DIR" ]; then
  echo "==> Copying NVIDIA keyring to ${KEYRING_DST}..."
  sudo mkdir -p "$KEYRING_DST"
  sudo cp "${LOCAL_REPO_DIR}/"cuda-*-keyring.gpg "$KEYRING_DST"/
else
  echo "Warning: expected local repo dir ${LOCAL_REPO_DIR} not found; continuing..." >&2
fi

# 4) Update and install toolkit
echo "==> Updating APT index..."
sudo apt-get update -y

if dpkg -s "$TOOLKIT_PKG" >/dev/null 2>&1; then
  echo "==> ${TOOLKIT_PKG} already installed. Skipping."
else
  echo "==> Installing ${TOOLKIT_PKG}..."
  sudo apt-get install -y "$TOOLKIT_PKG"
fi

# =============================
# Post-install cleanup
# =============================
echo "==> Cleaning installer artifacts..."

# Remove the local repo *package* so APT stops tracking the file: source
if dpkg -s "cuda-repo-wsl-ubuntu-${CUDA_SHORT}-local" >/dev/null 2>&1; then
  echo "==> Removing local repo package cuda-repo-wsl-ubuntu-${CUDA_SHORT}-local..."
  sudo dpkg -r "cuda-repo-wsl-ubuntu-${CUDA_SHORT}-local" || true
fi

# Remove local repo directory
echo "==> Removing local repo directory ${LOCAL_REPO_DIR}..."
sudo rm -rf "${LOCAL_REPO_DIR}" || true

# Purge any APT source entries that reference the removed local file repo
remove_local_repo_sources || true

# Routine APT tidy
echo "==> Autoremoving and cleaning APT cache..."
sudo apt-get -y autoremove
sudo apt-get -y clean

# Final update after cleanup (should NOT reference file: repo anymore)
echo "==> Final APT update after cleanup..."
sudo apt-get update -y

echo "==> Done. CUDA toolkit ${CUDA_SHORT} installed. Restart your WSL shell/session if needed."
