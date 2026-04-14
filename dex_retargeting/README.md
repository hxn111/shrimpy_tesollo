# Dex Retargeting
Source: https://github.com/dexsuite/dex-retargeting/blob/main/example/vector_retargeting/README.md

# Setup
```bash
# Setup virtual environment (bash):
python3.11 -m venv shrimp-venv
source venv/bin/activate

# Install dependencies
pip install -e dex_retargeting
pip install -r dex_retargeting/example/vector_retargeting/requirements.txt
pip install numpy==1.26.4  # Must be installed after everything in requirements.txt or causes issues (ignore red warning)
```