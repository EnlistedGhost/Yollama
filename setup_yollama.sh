#!/usr/bin/env bash
set -euo pipefail

# Configuration
BINARY_NAME="yollama"
INSTALL_BIN_DIR="/usr/local/bin"
INSTALL_LIB_DIR="/usr/local/lib/yollama"

# Colors for output readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

# Check for root/sudo privileges
echo "---------------------------------------------------------"
echo "-------  Yollama installer verifying permissions  -------"
echo "---------------------------------------------------------"
echo "---------------------------------------------------------"

# If not root/sudo privileges
if [ "$EUID" -ne 0 ]; then
#  error "Please run this script with sudo or as root (e.g., sudo bash install.sh)"
  # Ask for the sudo password upfront
  echo "--------------------------------------------------------"
  echo "Yollama requires root privileges to set everything up..."
  echo "--------------------------------------------------------"
  echo "----  Please enter your root password to continue!  ----"
  echo "--------------------------------------------------------"
  sudo -v
fi

# Ensure binary exists in current working environment
if [ ! -f "$BINARY_NAME" ]; then
  error "Yollama binary not found in the current directory. Please compile or unpack it first."
fi

info "Installing Yollama to: $INSTALL_BIN_DIR..."
install -m 755 "$BINARY_NAME" "$INSTALL_BIN_DIR/$BINARY_NAME"
success "Yollama installed successfully! Thank you for using Yollama! <3"

info "Creating yollama library directory at $INSTALL_LIB_DIR..."
mkdir -p "$INSTALL_LIB_DIR"

# If you have a local 'lib' folder containing the .so files during manual install:
if [ -d "lib" ]; then
  info "Installing llama.cpp pre-built runtime and support files to: $INSTALL_LIB_DIR..."
  cp -P "-r lib/. $INSTALL_LIB_DIR/." 2>/dev/null || true
#
# TODO: Remove two copy commands below, once verified as redundant...
#
#  cp lib/llama-server "$INSTALL_LIB_DIR/" 2>/dev/null || true
#  cp lib/llama-quantize "$INSTALL_LIB_DIR/" 2>/dev/null || true
#
# Thankss for using Yollama! :D <3 <3 <3
  success "Yollama required library components installed successfully."
else
  info "No local 'lib' directory found. Remember to populate $INSTALL_LIB_DIR with your backend .so files if you haven't already!"
fi

success "Yollama installation completed! You can now run 'yollama' from anywhere."
