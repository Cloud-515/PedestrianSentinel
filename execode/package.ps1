#Requires -Version 5.1
<#
.SYNOPSIS
    把已构建好的发布目录压成可分发的单文件 zip。

.DESCRIPTION
    build.ps1 的最后一步就是调用本脚本。也可以单独跑 —— 给一个已经构建好的
    dist\行人警戒区域监控\ 重新出包，不必重跑五分钟的 PyInstaller。

    为什么要出 zip：裸目录有四千多个文件、850 MB，走 U 盘或网盘时最容易的故障
    就是少传一个 _internal\ 里的 DLL，表现是「双击没反应」，排查起来最费劲。
    单个 zip 加一个 SHA256 就能把这类问题挡在门外。

    为什么用「白名单复制」而不是「原地删运行痕迹」：
      * 发布目录里的 logs\selftest.log 和 events\ 是验收证据，构建完还要能翻，
        不该为了出个包就被删掉；
      * 白名单漏不进东西。黑名单只挡得住已知的痕迹类型，以后程序多生成一种就会
        悄悄混进发布包 —— 而 config.json 里有构建机的绝对路径和一段本机视频的
        文件名，混进去就等于发出去了。
    代价是要一份临时副本（约 850 MB），出包后自动删除。

.EXAMPLE
    # 给现有的 dist\行人警戒区域监控\ 出包，不动原目录
    powershell -ExecutionPolicy Bypass -File .\execode\package.ps1

.EXAMPLE
    # 出包后保留临时副本，用来人工检查进包的到底是哪些文件
    powershell -ExecutionPolicy Bypass -File .\execode\package.ps1 -KeepStage
#>
[CmdletBinding()]
param(
    # 要打包的发布目录。默认 execode\dist\行人警戒区域监控。
    [string]$ReleaseDir = "",
    # zip 的输出目录。默认与发布目录同级（即 execode\dist）。
    [string]$OutDir = "",
    # 出包后保留临时副本，便于人工核对进包内容。
    [switch]$KeepStage
)

$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ReleaseName = "行人警戒区域监控"
$VersionFile = Join-Path $PSScriptRoot "version_info.txt"
$StageRoot   = Join-Path $PSScriptRoot "stage"

if (-not $ReleaseDir) { $ReleaseDir = Join-Path $PSScriptRoot "dist\$ReleaseName" }
if (-not $OutDir)     { $OutDir     = Split-Path -Parent $ReleaseDir }

# 进包的东西，逐项列明。新增要发布的文件必须同时加到这里，否则它进不了包 ——
# 下面的「意外条目」检查会当场报错，不会让你悄悄发出一个缺文件的版本。
$ShipItems = @(
    "$ReleaseName.exe",
    "_internal",
    "models",
    "assets",
    "使用说明.txt"
)

# 首次运行与自检会生成的运行痕迹，一律不进包。
$KnownResidue = @(
    "logs", "matplotlib", "ultralytics", "profiles",
    "events", "selftest-events", "config.json"
)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ">> $Message" -ForegroundColor Cyan
}

# 杀毒软件的实时扫描常攥着刚写完的 DLL 句柄，这时删除会得到 Access denied。
# 等一下再试即可，所以退避重试，而不是让整趟白跑。
function Remove-WithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Attempts = 10
    )
    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -Confirm:$false -ErrorAction Stop
            if ($i -gt 1) { Write-Host "   （第 $i 次尝试成功）" -ForegroundColor DarkGray }
            return
        } catch {
            if ($i -eq $Attempts) {
                throw ("删除失败（已重试 $Attempts 次）: $Path`n" +
                       "$($_.Exception.Message)`n" +
                       "常见原因：杀毒软件正在扫描这些 DLL，或有资源管理器/终端停留在该目录。")
            }
            Start-Sleep -Milliseconds (400 * $i)
        }
    }
}

function Invoke-Robocopy {
    param([Parameter(Mandatory = $true)][string]$Source,
          [Parameter(Mandatory = $true)][string]$Destination)
    robocopy $Source $Destination /E /MT:16 /R:2 /W:1 /NFL /NDL /NJH /NJS | Out-Null
    # robocopy 的退出码 0-7 都算成功（1 = 有文件被复制），>= 8 才是真失败。
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy 失败（退出码 $LASTEXITCODE）: $Source -> $Destination"
    }
}

function Get-SizeMB([string]$Path) {
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force |
            Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return [math]::Round($sum / 1MB, 1)
}

# --- 1. 检查发布目录 ------------------------------------------------------------
Write-Step "检查发布目录"
if (-not (Test-Path -LiteralPath $ReleaseDir -PathType Container)) {
    throw "找不到发布目录 $ReleaseDir。先跑一次 build.ps1。"
}
Write-Host "   $ReleaseDir"

foreach ($item in $ShipItems) {
    if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDir $item))) {
        throw "发布目录里缺少 $item，这个包发出去是坏的。先重跑 build.ps1。"
    }
}

# 既不在白名单、也不是已知运行痕迹的东西 —— 停下来问人。它要么是新增的发布内容
# （该加进 $ShipItems），要么是程序新生成的痕迹（该加进 $KnownResidue）。
# 两种都不该由脚本替人猜：猜错一次就是发出一个缺文件的包，或泄一份本机路径。
$unknown = Get-ChildItem -LiteralPath $ReleaseDir -Force |
           Where-Object { $ShipItems -notcontains $_.Name -and $KnownResidue -notcontains $_.Name }
if ($unknown) {
    throw ("发布目录里出现了没见过的条目：" + ($unknown.Name -join "、") + "`n" +
           "如果是要一起发布的内容，请加进 package.ps1 的 `$ShipItems；`n" +
           "如果是运行时生成的痕迹，请加进 `$KnownResidue。")
}

$excluded = Get-ChildItem -LiteralPath $ReleaseDir -Force |
            Where-Object { $KnownResidue -contains $_.Name }
if ($excluded) {
    Write-Host ("   排除运行痕迹（原目录不动）: " + ($excluded.Name -join "、")) -ForegroundColor DarkGray
}

# --- 2. 版本号 ------------------------------------------------------------------
# 从 version_info.txt 取，不在这儿再写一份 —— 两处版本号一定会漂移，而漂移的后果
# 是发出去的包名和 exe 属性面板里的版本号对不上。
$VersionText = Get-Content -LiteralPath $VersionFile -Raw -Encoding UTF8
if ($VersionText -match 'filevers=\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,') {
    $Version = "{0}.{1}.{2}" -f $Matches[1], $Matches[2], $Matches[3]
} else {
    throw "在 $VersionFile 里没找到 filevers=(x, y, z, w)，无法确定版本号"
}
$ArchivePath = Join-Path $OutDir ("{0}-v{1}-win64.zip" -f $ReleaseName, $Version)
Write-Host "   版本 v$Version -> $(Split-Path -Leaf $ArchivePath)"

# --- 3. 空间检查 ----------------------------------------------------------------
# 临时副本 + zip 都落在同一个盘上。空间不够的话，让它现在就报错，而不是复制到
# 一半失败后留下半个副本。
$SourceMB = Get-SizeMB $ReleaseDir
$driveName = (Get-Item -LiteralPath $OutDir).PSDrive.Name
$freeMB = [math]::Round((Get-PSDrive $driveName).Free / 1MB, 1)
$neededMB = $SourceMB * 1.6   # 副本 850 MB + zip 约 400 MB，留点余量
Write-Host ("   发布目录 {0} MB，{1}: 盘可用 {2} MB" -f $SourceMB, $driveName, $freeMB)
if ($freeMB -lt $neededMB) {
    throw ("空间不足：出包大约需要 {0} MB（临时副本 + zip），{1}: 盘只剩 {2} MB。" -f
           [math]::Round($neededMB), $driveName, $freeMB)
}

# --- 4. 白名单复制到临时副本 ----------------------------------------------------
# zip 内要带一层「行人警戒区域监控\」，所以临时副本的目录名必须就是它 ——
# CreateFromDirectory 的 includeBaseDirectory 取的是这个名字。
$StageDir = Join-Path $StageRoot $ReleaseName
Write-Step "准备临时副本（只复制要发布的内容）"
if (Test-Path -LiteralPath $StageRoot) { Remove-WithRetry -Path $StageRoot }
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

$stopwatch = [Diagnostics.Stopwatch]::StartNew()
foreach ($item in $ShipItems) {
    $source = Join-Path $ReleaseDir $item
    if (Test-Path -LiteralPath $source -PathType Container) {
        Invoke-Robocopy -Source $source -Destination (Join-Path $StageDir $item)
    } else {
        Copy-Item -LiteralPath $source -Destination (Join-Path $StageDir $item) -Force
    }
    Write-Host "   [ok] $item"
}
Write-Host ("   耗时 {0:N0} 秒" -f $stopwatch.Elapsed.TotalSeconds)

$stageMB = Get-SizeMB $StageDir
$stageFiles = (Get-ChildItem -LiteralPath $StageDir -Recurse -File -Force).Count
Write-Host ("   副本 {0} MB / {1} 个文件" -f $stageMB, $stageFiles)

# --- 5. 压缩 --------------------------------------------------------------------
Write-Step "压缩为 zip"
Add-Type -AssemblyName System.IO.Compression          # ZipArchive / ZipArchiveMode / CompressionLevel
Add-Type -AssemblyName System.IO.Compression.FileSystem  # ZipFile / ZipFileExtensions
if (Test-Path -LiteralPath $ArchivePath) { Remove-WithRetry -Path $ArchivePath }

# 逐条写入而不用 CreateFromDirectory：.NET Framework 那个方法会把条目名写成
# 反斜杠分隔（行人警戒区域监控\使用说明.txt），而 ZIP 规范（APPNOTE 4.4.17.1）
# 要求正斜杠。Windows 的资源管理器和 7-Zip 容得下，但 Linux/macOS 的 unzip 会把
# 反斜杠当成文件名里的普通字符 —— 四千个文件全被拍平到一个目录里。发出去的东西
# 不该赌对方用什么解压。
# 自己拼条目名还有个好处：顶层目录名是显式给的，不依赖 includeBaseDirectory
# 去取临时副本的目录名。
# 显式 UTF8 让中文文件名（exe 本身、使用说明.txt）在别人机器上不乱码 —— 不给这个
# 参数，.NET 会按 CP437 写名字，非中文区域设置的机器上解出来就是一串问号。
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$stageFull = (Resolve-Path -LiteralPath $StageDir).Path
$prefixLength = $stageFull.Length + 1
$archive = [System.IO.Compression.ZipFile]::Open(
    $ArchivePath,
    [System.IO.Compression.ZipArchiveMode]::Create,
    [System.Text.Encoding]::UTF8
)
try {
    foreach ($file in Get-ChildItem -LiteralPath $StageDir -Recurse -File -Force) {
        $relative = $file.FullName.Substring($prefixLength).Replace('\', '/')
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive,
            $file.FullName,
            "$ReleaseName/$relative",
            [System.IO.Compression.CompressionLevel]::Optimal
        )
    }
} finally {
    $archive.Dispose()
}
Write-Host ("   耗时 {0:N0} 秒" -f $stopwatch.Elapsed.TotalSeconds)

# --- 6. 回读校验 ----------------------------------------------------------------
# 压完就信它，等于把「包里到底有没有东西」这个问题留给拿到包的人去发现。
Write-Step "回读校验"
$zip = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
try {
    $entries = $zip.Entries | ForEach-Object { $_.FullName }
} finally {
    $zip.Dispose()
}

$expectedTop = "$ReleaseName/"
$badTop = $entries | Where-Object { -not $_.StartsWith($expectedTop) }
if ($badTop) {
    throw ("zip 里有条目不在「$ReleaseName/」下，解压会铺满当前目录：" +
           (($badTop | Select-Object -First 5) -join "、"))
}

# 中文条目名要能原样读回来。读回来是乱码，说明 UTF-8 标志没写上。
# 注意 ${} 不能省：汉字是 Unicode 字母类，属于合法标识符字符，写成
# "$expectedTop使用说明.txt" 会被解析成一个叫 $expectedTop使用说明 的变量。
$readmeEntry = "${expectedTop}使用说明.txt"
if ($entries -notcontains $readmeEntry) {
    throw "zip 里找不到条目 $readmeEntry —— 中文文件名可能没按 UTF-8 编码"
}
if ($entries -notcontains "$expectedTop$ReleaseName.exe") {
    throw "zip 里找不到 $ReleaseName.exe"
}

# 运行痕迹一个都不许在里面。
foreach ($name in $KnownResidue) {
    $leaked = $entries | Where-Object { $_ -eq "$expectedTop$name" -or $_.StartsWith("$expectedTop$name/") }
    if ($leaked) {
        throw "zip 里混进了运行痕迹 $name（$($leaked.Count) 个条目），里面可能有构建机的绝对路径"
    }
}

# 目录条目不单独写，所以 Entries 里全是文件，可以直接跟副本的文件数对。
$fileEntries = $entries.Count
if ($fileEntries -ne $stageFiles) {
    throw "zip 里有 $fileEntries 个文件，副本里有 $stageFiles 个，数量不符"
}
Write-Host ("   [ok] 顶层目录「$ReleaseName/」，{0} 个文件，中文名正常，无运行痕迹" -f $fileEntries)

# --- 7. 清理与报告 --------------------------------------------------------------
if (-not $KeepStage) {
    Write-Step "删除临时副本"
    Remove-WithRetry -Path $StageRoot
    Write-Host "   已删除 $StageRoot"
} else {
    Write-Host ""
    Write-Host "   临时副本保留在 $StageDir（-KeepStage）" -ForegroundColor DarkGray
}

$archiveMB = [math]::Round((Get-Item -LiteralPath $ArchivePath).Length / 1MB, 1)
# 几百 MB 的包多半要过一次网盘或 U 盘，附个校验值，对方能自己确认没传坏 ——
# 半个 DLL 损坏的表现是「双击没反应」，那种问题排查起来最费劲。
$archiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash

Write-Step "完成"
Write-Host "   可交付的分发包（把这一个文件发出去）:" -ForegroundColor Green
Write-Host ("   {0}" -f $ArchivePath) -ForegroundColor Green
Write-Host ("   {0} MB（解压后 {1} MB，压缩率 {2:P0}）" -f
            $archiveMB, $stageMB, (1 - $archiveMB / $stageMB))
Write-Host ("   SHA256: {0}" -f $archiveHash) -ForegroundColor DarkGray
