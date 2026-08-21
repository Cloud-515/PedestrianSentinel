#Requires -Version 5.1
<#
.SYNOPSIS
    行人警戒区域监控 —— 单目录（onedir）绿色版构建脚本。

.DESCRIPTION
    产出两样东西：
      execode\dist\行人警戒区域监控\              可整个目录拷走即用
      execode\dist\行人警戒区域监控-vX.Y.Z-win64.zip  对外发布用的单文件分发包

    体积的关键在于用一个**只装本程序所需库**的专用虚拟环境来打包（-Setup），
    尤其是 torch 走 CPU 索引 —— 装 +cu121 的话 torch\lib 一家就 4.7 GB。

.EXAMPLE
    # 首次：建 .venv-build 并构建（会下载几百 MB，耗时数分钟）
    powershell -ExecutionPolicy Bypass -File .\execode\build.ps1 -Setup

.EXAMPLE
    # 日常：复用已有 .venv-build，只重新构建
    powershell -ExecutionPolicy Bypass -File .\execode\build.ps1

.EXAMPLE
    # 发布前的完整验收：连真实推理一起验（载权重、推真帧、验告警落盘、放告警音）
    powershell -ExecutionPolicy Bypass -File .\execode\build.ps1 -DeepTestVideo ".\8月11日-1.mp4"
#>
[CmdletBinding()]
param(
    # 创建并安装打包专用虚拟环境。已存在则原地补齐依赖。
    [switch]$Setup,
    # 跳过打包后的 --selftest 验收（仅调试构建流程时用）。
    [switch]$SkipSelfTest,
    # 跳过最后的 zip 打包。压 873 MB 要几分钟，反复调构建流程时可以省掉。
    [switch]$SkipPackage,
    # 给了视频就在自检里加跑 --deep：真的载权重、推真帧、验告警落盘。
    # 不给则只做默认的快速自检（import 模块 + 查资源），验不到推理能不能跑。
    [string]$DeepTestVideo = "",
    # 用来创建 venv 的基础解释器，需 >= 3.10。
    [string]$BasePython = "E:\python\python.exe"
)

$ErrorActionPreference = "Stop"

# 从源头掐掉中文乱码：让子进程用 UTF-8，也让本进程按 UTF-8 解码子进程输出。
# execode\build\琛屼汉璀︽垝鍖哄煙鐩戞帶 那个乱码目录就是没做这件事的产物。
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# 参数校验放在构建**之前**：PyInstaller 那一步要跑五分钟，视频路径打错却要等到
# 第 7 步才发现，白烧一次构建。这里就地解析成绝对路径，后面直接用。
$DeepTestVideoPath = ""
if ($DeepTestVideo) {
    if (-not (Test-Path -LiteralPath $DeepTestVideo -PathType Leaf)) {
        throw "找不到 -DeepTestVideo 指定的视频: $DeepTestVideo"
    }
    $DeepTestVideoPath = (Resolve-Path -LiteralPath $DeepTestVideo).Path
}

$VenvDir     = Join-Path $ProjectRoot ".venv-build"
$VenvPython  = Join-Path $VenvDir "Scripts\python.exe"
$SpecFile    = Join-Path $PSScriptRoot "PedestrianZoneMonitor.spec"
$VersionFile = Join-Path $PSScriptRoot "version_info.txt"
$WorkPath    = Join-Path $PSScriptRoot "build"
$DistPath    = Join-Path $PSScriptRoot "dist"

$BuildName   = "PedestrianZoneMonitor"      # spec 里的 ASCII 名字
$ReleaseName = "行人警戒区域监控"            # 最终对外的名字
$BuildDir    = Join-Path $DistPath $BuildName
$ReleaseDir  = Join-Path $DistPath $ReleaseName

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ">> $Message" -ForegroundColor Cyan
}

function Get-SizeMB([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force |
            Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return [math]::Round($sum / 1MB, 1)
}

function Invoke-Pip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PipArgs)
    & $VenvPython -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip $($PipArgs -join ' ') 失败, 退出码 $LASTEXITCODE"
    }
}

# COLLECT 刚写完近千个 DLL，杀毒软件的实时扫描往往还攥着句柄，这时候对它们改名或
# 删除会得到 "Access to the path ... is denied"。等一下再试即可，所以文件系统操作
# 统一走退避重试，而不是让一次五分钟的构建白跑。
# 实测印证：dist 目录改名当场失败，隔几分钟手动重试第一次就成 —— 纯粹的瞬时占用。
function Rename-WithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$NewName,
        [int]$Attempts = 10
    )
    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            Rename-Item -LiteralPath $Path -NewName $NewName -ErrorAction Stop
            if ($i -gt 1) { Write-Host "   （第 $i 次尝试成功）" -ForegroundColor DarkGray }
            return
        } catch {
            if ($i -eq $Attempts) {
                throw ("改名失败（已重试 $Attempts 次）: $Path -> $NewName`n" +
                       "$($_.Exception.Message)`n" +
                       "常见原因：杀毒软件正在扫描刚生成的 DLL，或有资源管理器/终端停留在该目录。")
            }
            Start-Sleep -Milliseconds (400 * $i)
        }
    }
}

function Remove-WithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Attempts = 10
    )
    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            if ($i -gt 1) { Write-Host "   （第 $i 次尝试成功）" -ForegroundColor DarkGray }
            return
        } catch {
            if ($i -eq $Attempts) {
                throw ("删除失败（已重试 $Attempts 次）: $Path`n" +
                       "$($_.Exception.Message)`n" +
                       "常见原因：杀毒软件正在扫描上一轮生成的 DLL，或有资源管理器/终端停留在该目录。")
            }
            Start-Sleep -Milliseconds (400 * $i)
        }
    }
}

# --- 1. 打包专用虚拟环境 --------------------------------------------------------
if ($Setup) {
    Write-Step "准备打包专用虚拟环境 .venv-build"
    if (-not (Test-Path -LiteralPath $BasePython)) {
        throw "找不到基础解释器 $BasePython，请用 -BasePython 指定一个 >= 3.10 的 python.exe"
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        & $BasePython -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败, 退出码 $LASTEXITCODE" }
    } else {
        Write-Host "   已存在，跳过创建，仅补齐依赖"
    }

    Invoke-Pip install --upgrade pip wheel

    # 关键一步：CPU 版 torch。默认 PyPI 装的是 +cu121，torch\lib 里 4.7 GB 全是
    # 本程序用不到的 CUDA DLL。这个索引同时也托管 torch 自己的依赖，可单独使用。
    Write-Host "   [1/4] torch CPU 版（几百 MB，慢）" -ForegroundColor DarkGray
    Invoke-Pip install torch==2.5.1 torchvision==0.20.1 `
        --index-url https://download.pytorch.org/whl/cpu

    # 只要 Essentials：PySide6 元包会连带装 PySide6_Addons
    #（Qt3D / Charts / WebEngine / DataVisualization，几百 MB），本程序只用
    # QtCore/QtGui/QtWidgets。装 Essentials 后 `import PySide6` 照常可用。
    Write-Host "   [2/4] PySide6_Essentials" -ForegroundColor DarkGray
    Invoke-Pip install PySide6_Essentials==6.11.1

    Write-Host "   [3/4] 推理与打包依赖" -ForegroundColor DarkGray
    Invoke-Pip install numpy supervision trackers ultralytics openvino==2024.6.0 pyinstaller

    # ultralytics 的元数据按名字要求 opencv-python，所以 headless 只能事后替换：
    # 先卸掉带 GUI 的版本，再用 --no-deps 装 headless（否则 pip 会把它解析回来）。
    # 本程序的 cv2 只用 VideoCapture/imwrite/绘图，cv2.imshow 仅出现在未被 GUI
    # 引用的 security_monitor.py 里。headless 版更小，也不会带一套自己的 Qt 插件
    # 来和 PySide6 抢。
    Write-Host "   [4/4] 换用 opencv-python-headless" -ForegroundColor DarkGray
    & $VenvPython -m pip uninstall -y opencv-python opencv-contrib-python
    Invoke-Pip install --no-deps opencv-python-headless

    Write-Host "   虚拟环境就绪" -ForegroundColor Green
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "找不到 $VenvPython。首次构建请加 -Setup 参数。"
}
if (-not (Test-Path -LiteralPath $SpecFile)) {
    throw "找不到 spec 文件 $SpecFile"
}
# 版本资源文件缺了的话，PyInstaller 要跑到快结束才抱怨，而 package.ps1 也指着它取
# 包名里的版本号。放在这儿一并挡掉，别等五分钟。
if (-not (Test-Path -LiteralPath $VersionFile)) {
    throw "找不到版本资源文件 $VersionFile（spec 与 package.ps1 都需要它）"
}

# --- 2. 清理 --------------------------------------------------------------------
Write-Step "清理旧的 build/dist"
foreach ($stale in @($WorkPath, $DistPath)) {
    if (Test-Path -LiteralPath $stale) {
        Remove-WithRetry -Path $stale
        Write-Host "   已删除 $stale"
    }
}

# --- 3. PyInstaller -------------------------------------------------------------
Write-Step "调用 PyInstaller（单目录 onedir 模式）"
& $VenvPython -m PyInstaller $SpecFile --noconfirm --distpath $DistPath --workpath $WorkPath
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败, 退出码 $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $BuildDir)) { throw "PyInstaller 未产出 $BuildDir" }

# --- 4. 拷外部资源 --------------------------------------------------------------
# 这些**刻意不进** spec 的 datas：放在 _internal\ 外面用户才能直接替换。
Write-Step "拷贝可替换资源（models\ 与 assets\）"
$ModelsDir = Join-Path $BuildDir "models"
$AssetsDir = Join-Path $BuildDir "assets"
New-Item -ItemType Directory -Force -Path $ModelsDir, $AssetsDir | Out-Null

$Payload = @(
    @{ Src = Join-Path $ProjectRoot "yolo11n.pt";                        Dst = $ModelsDir; Label = "models\yolo11n.pt" }
    @{ Src = Join-Path $ProjectRoot "models\yolo11n_int8_openvino_model"; Dst = $ModelsDir; Label = "models\yolo11n_int8_openvino_model\" }
    @{ Src = Join-Path $ProjectRoot "warming_converted.wav";             Dst = $AssetsDir; Label = "assets\warming_converted.wav" }
)
foreach ($item in $Payload) {
    if (Test-Path -LiteralPath $item.Src) {
        Copy-Item -LiteralPath $item.Src -Destination $item.Dst -Recurse -Force
        Write-Host "   [ok] $($item.Label)"
    } else {
        # 缺资源不该让构建失败：模型和音频都是运行期才需要，事后补进去即可。
        Write-Warning "   缺失: $($item.Src) -> 发布目录里将没有 $($item.Label)"
    }
}

# --- 5. 使用说明 ----------------------------------------------------------------
Write-Step "生成使用说明.txt"
$Readme = @'
行人警戒区域监控 —— 单目录绿色版

一、运行
    双击「行人警戒区域监控.exe」。首次运行会在本目录生成 config.json、
    profiles\、events\、logs\，属正常现象。

二、目录说明
    _internal\   程序运行库，请勿删除或改名，否则无法启动
    models\
      yolo11n.pt                    默认检测模型，可替换为同格式的其他 YOLO 权重
      yolo11n_int8_openvino_model\  「CPU 低功耗模式」专用，缺失则该模式不可用
    assets\
      warming_converted.wav         告警提示音，可替换为任意 16bit PCM WAV
    events\      闯入取证：alarm_events.jsonl 与 screenshots\
    profiles\    警戒区配置组
    logs\        运行日志 app.log，反馈问题时请一并提供

三、注意事项
    1. 整个文件夹要一起拷贝，单独复制 exe 无法运行。
    2. 本版本为 CPU 版，设备下拉框只显示 CPU，属预期行为。
    3. 若程序放在 C:\Program Files 等只读位置，配置与取证记录会自动改存到
       %LOCALAPPDATA%\行人警戒区域监控\，日志里会有一行对应提示。
    4. 诊断用自检（不开窗口，结果写 logs\selftest.log）：
           行人警戒区域监控.exe --selftest
'@
# UTF-8 带 BOM：让双击用记事本打开时也能正确识别中文。
[System.IO.File]::WriteAllText(
    (Join-Path $BuildDir "使用说明.txt"),
    $Readme,
    (New-Object System.Text.UTF8Encoding($true))
)

# --- 6. 改回中文名 --------------------------------------------------------------
# 安全性已从 PyInstaller 源码确认：contents_directory("_internal") 是写在 exe
# TOC 里的 OPTION 项，引导程序按 **exe 所在目录** 解析它，与 exe 文件名无关。
Write-Step "改名为「$ReleaseName」"
Rename-WithRetry -Path (Join-Path $BuildDir "$BuildName.exe") -NewName "$ReleaseName.exe"
Rename-WithRetry -Path $BuildDir -NewName $ReleaseName
$ReleaseExe = Join-Path $ReleaseDir "$ReleaseName.exe"
if (-not (Test-Path -LiteralPath $ReleaseExe)) { throw "改名后找不到 $ReleaseExe" }

# --- 7. 自检验收 ----------------------------------------------------------------
# 窗口化 exe（console=False）没有 stdout 可看，所以 --selftest 把报告写文件，
# 这里读回来打印。跑的是**改名后**的 exe，顺带验证第 6 步没把引导搞坏。
if (-not $SkipSelfTest) {
    Write-Step "运行打包版自检（--selftest）"
    $SelfTestLog = Join-Path $ReleaseDir "logs\selftest.log"
    if (Test-Path -LiteralPath $SelfTestLog) { Remove-Item -LiteralPath $SelfTestLog -Force }

    # 刻意不用 -Wait：窗口化 exe 若在自检里崩到 PyInstaller 的模态 traceback 对话框，
    # -Wait 会一直等人点「Close」，无人值守的构建就此挂死（曾经真挂住过）。
    # main.py 的 _run_selftest 现在每一步都挡异常，正常不该再出现那个弹窗 —— 但构建
    # 脚本不该把「不该发生」当成保证，所以给个上限，超时就杀掉并说清怀疑对象。
    $SelfTestTimeoutSec = 180
    $SelfTestArgs = @("--selftest")
    if ($DeepTestVideoPath) {
        # 路径可能带空格，Start-Process 的 -ArgumentList 只做空格拼接、不补引号。
        $SelfTestArgs += @("--deep", "--video", ('"{0}"' -f $DeepTestVideoPath))
        # 深度自检要载两次权重（含 OpenVINO）、推几百帧、还同步放一遍告警音，
        # 180 秒远远不够；CPU 上实测数分钟量级。
        $SelfTestTimeoutSec = 900
        Write-Host "   深度验收已开启（真实推理 + 告警落盘 + 告警音）" -ForegroundColor Yellow
        Write-Host "   素材: $DeepTestVideoPath"
    }
    $proc = Start-Process -FilePath $ReleaseExe -ArgumentList $SelfTestArgs -PassThru
    if (-not $proc.WaitForExit($SelfTestTimeoutSec * 1000)) {
        try { Stop-Process -Id $proc.Id -Force -ErrorAction Stop } catch { }
        throw ("自检超过 $SelfTestTimeoutSec 秒未退出，已强制结束。`n" +
               "常见原因：自检中弹出了未捕获异常对话框，正等人点确认。`n" +
               "请查看 $SelfTestLog 与 $(Join-Path $ReleaseDir 'logs\app.log')。")
    }

    if (Test-Path -LiteralPath $SelfTestLog) {
        Get-Content -LiteralPath $SelfTestLog -Encoding UTF8 | ForEach-Object { Write-Host "   $_" }
    } else {
        Write-Warning "   自检没有产出 $SelfTestLog（可能启动即崩，检查 logs\app.log）"
    }
    if ($proc.ExitCode -ne 0) {
        throw "打包版自检失败, 退出码 $($proc.ExitCode)。详见上面的报告与 logs\app.log"
    }
    Write-Host "   自检通过" -ForegroundColor Green
}

# --- 8. 体积报告 ----------------------------------------------------------------
# 先报体积再出包，好让 package.ps1 的「可交付的分发包」那段留在屏幕最后一屏。
Write-Step "构建完成"
$internalMB = Get-SizeMB (Join-Path $ReleaseDir "_internal")
$totalMB    = Get-SizeMB $ReleaseDir
Write-Host ("   输出目录 : {0}" -f $ReleaseDir)
Write-Host ("   _internal: {0} MB" -f $internalMB)
Write-Host ("   总计     : {0} MB" -f $totalMB) -ForegroundColor Green

# --- 9. 压成分发包 --------------------------------------------------------------
# 到上一步为止，产物是个 870 MB 上下、四千多个文件的裸目录 —— 那不是可交付物：
# 拷给别人的路上少一个 _internal\ 里的 DLL 就启动不了，而且谁也不会去翻 dist\。
# 压成单个 zip 才是能发出去的东西。
#
# 打包逻辑全在 package.ps1 里，这儿只负责调它。分成两个脚本是有原因的：出包要重跑
# 时（换版本号、上一次压坏了）不该再等一次五分钟的 PyInstaller。
# 另外注意它是「复制干净的一份再压」，不动 $ReleaseDir —— 所以上一步自检留下的
# logs\selftest.log 和 events\ 仍在原处可查，而 config.json（里面有构建机的绝对
# 路径和本机视频文件名）不会进包。
if (-not $SkipPackage) {
    # 用 & 而不是点源：package.ps1 里也有个 Remove-WithRetry，点源会覆盖掉本脚本的。
    & (Join-Path $PSScriptRoot "package.ps1") -ReleaseDir $ReleaseDir -OutDir $DistPath
} else {
    Write-Host ""
    Write-Host "   （已跳过 zip 打包：-SkipPackage）" -ForegroundColor DarkGray
    Write-Host "   需要出包时可单独跑：.\execode\package.ps1" -ForegroundColor DarkGray
}
