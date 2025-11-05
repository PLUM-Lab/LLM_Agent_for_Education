# Test OpenAI API Key (read from environment variable)
$apiKey = $env:OPENAI_API_KEY
if (-not $apiKey) {
    Write-Host "Please set OPENAI_API_KEY in your environment." -ForegroundColor Yellow
    Write-Host "Example (PowerShell):`n$env:OPENAI_API_KEY='sk-xxxx'" -ForegroundColor Yellow
    exit 1
}

$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer $apiKey"
}

$body = @{
    model = "gpt-3.5-turbo"
    messages = @(
        @{
            role = "user"
            content = "Hello, this is a test. Please respond with 'API Key is working!'"
        }
    )
    max_tokens = 50
} | ConvertTo-Json -Depth 10

Write-Host "Testing OpenAI API Key..." -ForegroundColor Yellow
Write-Host "API Key (last 4 chars): ...$($apiKey.Substring($apiKey.Length - 4))" -ForegroundColor Cyan
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "https://api.openai.com/v1/chat/completions" -Method Post -Headers $headers -Body $body -ErrorAction Stop
    
    Write-Host "✅ SUCCESS! API Key is working!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Cyan
    Write-Host $response.choices[0].message.content -ForegroundColor White
    Write-Host ""
    Write-Host "Model used: $($response.model)" -ForegroundColor Gray
    Write-Host "Tokens used: $($response.usage.total_tokens)" -ForegroundColor Gray
} catch {
    Write-Host "❌ ERROR: API Key test failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error details:" -ForegroundColor Yellow
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body:" -ForegroundColor Yellow
        Write-Host $responseBody -ForegroundColor Red
    }
}

