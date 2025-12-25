# GitHub Actions Setup Script for Windows
# Run this to initialize Git and prepare for deployment

Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host "  Face Mask Detection - GitHub Actions Setup" -ForegroundColor Cyan
Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "Checking for Git..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✓ $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Initialize Git repository
Write-Host "`nInitializing Git repository..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
} else {
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
}

# Create .gitattributes for proper line endings
Write-Host "`nConfiguring Git attributes..." -ForegroundColor Yellow
@"
* text=auto
*.py text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.md text eol=lf
*.sh text eol=lf
*.h5 binary
*.pb binary
*.tflite binary
"@ | Out-File -FilePath ".gitattributes" -Encoding utf8
Write-Host "✓ Created .gitattributes" -ForegroundColor Green

# Check if models exist
Write-Host "`nChecking for trained model..." -ForegroundColor Yellow
if (Test-Path "models/saved_model/saved_model.pb") {
    Write-Host "✓ Trained model found" -ForegroundColor Green
} else {
    Write-Host "⚠ Trained model not found!" -ForegroundColor Yellow
    Write-Host "  Run training first or model will need to be uploaded separately" -ForegroundColor Gray
}

# Add all files
Write-Host "`nStaging files for commit..." -ForegroundColor Yellow
git add .
Write-Host "✓ Files staged" -ForegroundColor Green

# Show status
Write-Host "`nGit Status:" -ForegroundColor Yellow
git status --short

# Create initial commit
Write-Host "`n===========================================================================" -ForegroundColor Cyan
Write-Host "Ready to commit!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run the following commands:" -ForegroundColor Green
Write-Host ""
Write-Host "  git commit -m 'Initial commit: Face mask detection system with CI/CD'" -ForegroundColor White
Write-Host "  git branch -M main" -ForegroundColor White
Write-Host "  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git" -ForegroundColor White
Write-Host "  git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Create a GitHub repository" -ForegroundColor Gray
Write-Host "2. Add GitHub Secrets (see DEPLOYMENT.md)" -ForegroundColor Gray
Write-Host "3. Push code to trigger GitHub Actions" -ForegroundColor Gray
Write-Host "4. Choose deployment platform (Streamlit Cloud / Heroku / Docker)" -ForegroundColor Gray
Write-Host ""
Write-Host "For detailed instructions, see: DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host ""
