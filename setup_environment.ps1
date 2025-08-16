# python check
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found!"
    exit 1
}

Write-Host "Creating virtual environment"
python -m venv gans-env

Write-Host "Activating virtual environment"
.\gans-env\Scripts\Activate.ps1

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Installing packages"
pip install -r requirements.txt

Write-Host "Done! Environment is active." -ForegroundColor Green
