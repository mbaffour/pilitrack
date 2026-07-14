#!/bin/bash
# macOS / Linux double-click launcher for the pilitrack browser app.
# First time on macOS: right-click -> Open (to bypass Gatekeeper), and if needed
# make it executable once with:  chmod +x "Start pilitrack (browser).command"
cd "$(dirname "$0")" || exit 1
# Activate a conda base if present so pilitrack-web is on PATH.
for A in "$HOME/miniconda3/bin/activate" "$HOME/anaconda3/bin/activate" \
         "$HOME/opt/miniconda3/bin/activate" "$HOME/opt/anaconda3/bin/activate"; do
  [ -f "$A" ] && source "$A" && break
done
echo "Starting pilitrack in your browser (keep this window open; close to stop)..."
pilitrack-web || {
  echo ""
  echo "Could not start automatically. Open a terminal and run:  pilitrack-web"
  read -n 1 -s -r -p "Press any key to close."
}
