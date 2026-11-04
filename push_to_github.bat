@echo off
cd /d "%~dp0"
title Push NebulaMapper to GitHub (GalaxyFSRP1/NebulaMapper)
echo ========================================================
echo   Push NebulaMapper to GitHub
echo   Target: https://github.com/GalaxyFSRP1/NebulaMapper.git
echo ========================================================
echo.
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)
set "REPO_URL=https://github.com/GalaxyFSRP1/NebulaMapper.git"
set /p "USER_URL=Press ENTER to push to %REPO_URL% (or paste another URL): "
if not "%USER_URL%"=="" set "REPO_URL=%USER_URL%"

if exist ".git" rmdir /s /q ".git"
git init
git config user.name "GalaxyFSRP1"
git config user.email "GalaxyFSRP1@users.noreply.github.com"
git branch -M main
git add -A
git commit -m "Release NebulaMapper - PhantomVoid Neural BeatStudio"
if errorlevel 1 (
    echo [ERROR] git commit failed.
    pause
    exit /b 1
)
git remote add origin "%REPO_URL%"
echo.
echo Pushing to %REPO_URL% ...
git push -u origin main --force
if errorlevel 1 (
    echo.
    echo [ERROR] git push failed. Make sure you are signed into GitHub
    echo and that https://github.com/GalaxyFSRP1/NebulaMapper exists.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   SUCCESS! Your repo is live at:
echo   https://github.com/GalaxyFSRP1/NebulaMapper
echo.
echo   One-Click Google Colab GPU Link:
echo   https://colab.research.google.com/github/GalaxyFSRP1/NebulaMapper/blob/main/colab.ipynb
echo ========================================================
pause
