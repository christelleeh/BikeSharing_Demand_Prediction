#!/bin/bash
# setup.sh
# Create venv → Install → Run notebook → Launch Streamlit

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip and installing requirements..."
python.exe -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Launching Streamlit app (Group_5_StreamlitSrc_Assigment2.py)..."
streamlit run "Group_5_StreamlitSrc_Assigment2.py"
