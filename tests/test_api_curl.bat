@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 测试 OpenAI API (curl)...
for /f "delims=" %%a in ('powershell -NoProfile -Command "$c=Get-Content api-key.js -Raw; $m=[regex]::Match($c, \"OPENAI_API_KEY\s*[=:]\s*['\`\"]([^'\`\"]+)['\`\"]\"); $m.Groups[1].Value"') do set KEY=%%a
if "%KEY%"=="" (
    echo [失败] 未找到 api-key.js 中的密钥
    exit /b 1
)
curl.exe -s -X POST "https://api.openai.com/v1/chat/completions" -H "Authorization: Bearer %KEY%" -H "Content-Type: application/json" -d @tests\test_curl_body.json
echo.
pause
