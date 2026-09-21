<#
.SYNOPSIS
    跑一遍静态检查与测试。

.DESCRIPTION
    与 CI（.github/workflows/tests.yml）做的两件事一致：ruff check + pytest。
    本机已验证可用的解释器是 .venv-openvino\Scripts\python.exe —— 项目根目录**没有**
    .venv；.venv-build 是 PyInstaller 打包环境，里面没装 pytest，别用它跑测试。

.EXAMPLE
    .\run_tests.ps1
    .\run_tests.ps1 -Python "E:\python\python.exe"
#>
param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not $Python) {
    $candidates = @(
        (Join-Path $root ".venv-openvino\Scripts\python.exe"),
        (Join-Path $root ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { $Python = $candidate; break }
    }
    if (-not $Python) {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($command) { $Python = $command.Source }
    }
}
if (-not $Python) {
    throw "找不到 python.exe，请用 -Python 指定解释器路径"
}
Write-Host "解释器: $Python" -ForegroundColor DarkGray

$failed = @()

Write-Host "`n[1/2] ruff check ." -ForegroundColor Cyan
& $Python -m ruff check . --no-cache
if ($LASTEXITCODE -eq 0) {
    Write-Host "  通过" -ForegroundColor Green
} elseif ($LASTEXITCODE -eq 1) {
    $failed += "ruff"
} else {
    # 退出码 2 表示 ruff 自己没跑起来（多半是没装）。
    Write-Warning "  跳过：当前环境没有 ruff。安装：$Python -m pip install ruff"
}

Write-Host "`n[2/2] pytest -q" -ForegroundColor Cyan
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { $failed += "pytest" }

if ($failed.Count -gt 0) {
    Write-Host "`n失败: $($failed -join '、')" -ForegroundColor Red
    exit 1
}
Write-Host "`n全部通过" -ForegroundColor Green
exit 0
