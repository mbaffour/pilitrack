#!/bin/bash
# macOS / Linux one-click installer for pilitrack.
# First time on macOS: right-click -> Open. If it won't run, make it executable:
#   chmod +x "Install pilitrack.command"
cd "$(dirname "$0")" || exit 1
for A in "$HOME/miniconda3/bin/activate" "$HOME/anaconda3/bin/activate" \
         "$HOME/opt/miniconda3/bin/activate" "$HOME/opt/anaconda3/bin/activate"; do
  [ -f "$A" ] && source "$A" && break
done
echo "Installing pilitrack and its dependencies (a few minutes on first run)..."
python -m pip install --upgrade pip
python -m pip install -e "pilitrack_pkg[all]" || {
  echo ""
  echo "Install failed. If you have no conda, install Miniconda first:"
  echo "  https://docs.conda.io/en/latest/miniconda.html"
  read -n 1 -s -r -p "Press any key to close."
  exit 1
}
echo ""
echo "All set! Double-click 'Start pilitrack (browser)' to launch."
read -n 1 -s -r -p "Press any key to close."
