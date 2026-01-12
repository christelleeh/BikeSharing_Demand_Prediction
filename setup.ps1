# setup.ps1
# Create venv → Install → Run notebook → Launch Streamlit

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing requirements..." -ForegroundColor Cyan
python.exe -m pip install --upgrade pip
pip install -r requirements.txt


Write-Host "Launching Streamlit app (Group_5_StreamlitSrc_Assigment2.py)..." -ForegroundColor Cyan
streamlit run "Group_5_StreamlitSrc_Assigment2.py"
