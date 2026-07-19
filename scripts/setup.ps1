<#
.SYNOPSIS
    Athena setup — prepares a freshly cloned repository for first run.

.DESCRIPTION
    Takes a machine with nothing installed and leaves it able to run Athena:

      1. Finds a supported Python (3.10-3.13), installing 3.12 if none exists.
      2. Creates the .venv virtual environment.
      3. Installs the Python dependencies from requirements.txt.
      4. Downloads a prebuilt llama.cpp runtime matching the detected GPU.
      5. Creates the models/reason and models/learning folders.

    Nothing is compiled and no build tools are needed. Every step is
    idempotent: re-running repairs a partial install rather than starting over.

.PARAMETER Backend
    Which prebuilt llama.cpp runtime to download.

      auto    (default) Detect the GPU and choose: Vulkan for AMD/Intel,
              CUDA for NVIDIA, CPU otherwise.
      vulkan  Vulkan offload. Works on AMD, Intel and NVIDIA. ~32 MB.
      cuda    CUDA offload. NVIDIA only, and a much larger download.
      cpu     No GPU offload. Always works, slowest. ~17 MB.
      none    Skip the runtime entirely — for use with an external provider
              such as LM Studio. See docs/SETUP.md.

.PARAMETER LlamaBuild
    Which llama.cpp release to install. Defaults to a specific build that has
    been verified against Athena. Pass "latest" to take the newest release.

.PARAMETER Dev
    Also install the development dependencies from requirements-dev.txt, so
    the test suite can be run.

.PARAMETER Force
    Re-download the llama.cpp runtime even if one is already installed.

.PARAMETER Yes
    Do not pause for confirmation before large downloads. Intended for
    unattended installs.

.NOTES
    Why a prebuilt runtime rather than llama-cpp-python
    ---------------------------------------------------
    llama-cpp-python publishes no prebuilt Windows wheels, so installing it
    means compiling llama.cpp locally: several GB of Visual Studio build tools
    and a 20-30 minute build. llama.cpp itself ships ready-made binaries for
    every backend, so Athena downloads one of those instead and talks to it
    over loopback HTTP. Same models, same speed, no compiler.

    Advanced users who prefer in-process inference can still install
    llama-cpp-python by hand and set provider to "llamacpp"; docs/SETUP.md
    explains how.
#>

[CmdletBinding()]
param(
    [ValidateSet('auto', 'vulkan', 'cuda', 'cpu', 'none')]
    [string]$Backend = 'auto',

    [string]$LlamaBuild = 'b10068',

    [switch]$Dev,

    [switch]$Force,

    [switch]$Yes
)

$ErrorActionPreference = 'Stop'

# Always operate on the repository root, regardless of where we were invoked.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Python versions Athena runs on. The floor is 3.10 because the code uses PEP
# 604 unions (`str | None`). The ceiling is the newest version the pinned
# dependencies publish wheels for.
$SupportedPython = @('3.12', '3.13', '3.11', '3.10')   # preference order
$PythonToInstall = '3.12'
$WingetPythonId = 'Python.Python.3.12'

$LlamaRepo = 'ggml-org/llama.cpp'
$RuntimeDir = Join-Path $RepoRoot 'runtime\llama'


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

function Write-Step { param($Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Ok { param($Text) Write-Host "    OK  $Text" -ForegroundColor Green }
function Write-Info { param($Text) Write-Host "    $Text" -ForegroundColor Gray }
function Write-Warn { param($Text) Write-Host "    !   $Text" -ForegroundColor Yellow }


# ---------------------------------------------------------------------------
# Python discovery and installation
# ---------------------------------------------------------------------------

function Test-PythonCandidate {
    <#
        Return the version string of a python.exe if it is a supported
        interpreter, otherwise $null.

        Guards against the Windows Store stub: an alias in WindowsApps that
        opens the Store instead of running Python.

        Uses `--version` rather than `-c` on purpose: Windows PowerShell
        mangles quote characters when forwarding arguments to a native
        executable, so any `-c` snippet containing quotes arrives corrupted.
    #>
    param([string]$Exe)

    if (-not $Exe -or -not (Test-Path $Exe)) { return $null }
    if ($Exe -like '*\WindowsApps\*') { return $null }

    try {
        $raw = & $Exe --version
    } catch {
        return $null
    }

    if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }

    if ("$raw".Trim() -match '^Python\s+(\d+\.\d+)') {
        $version = $Matches[1]
        if ($SupportedPython -contains $version) { return $version }
    }

    return $null
}

function Find-Python {
    <#
        Locate a supported interpreter already on this machine.

        Probed in preference order (3.12 first, then 3.13, 3.11, 3.10) so a
        machine with several Pythons gets the best-supported one rather than
        whichever happens to be first on PATH.

        Returns a hashtable @{ Exe; Version } or $null.
    #>
    $candidates = [System.Collections.ArrayList]::new()

    # The py launcher knows about every registered install.
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        foreach ($version in $SupportedPython) {
            try {
                $found = & py.exe "-$version" -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and $found) {
                    [void]$candidates.Add("$found".Trim())
                }
            } catch { }
        }
    }

    # Default install locations, used when winget has just installed Python and
    # PATH has not been refreshed in this process yet.
    foreach ($version in $SupportedPython) {
        $tag = $version.Replace('.', '')
        [void]$candidates.Add("$env:LOCALAPPDATA\Programs\Python\Python$tag\python.exe")
        [void]$candidates.Add("$env:ProgramFiles\Python$tag\python.exe")
    }

    # Whatever is on PATH, last.
    $onPath = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($onPath) { [void]$candidates.Add($onPath.Source) }

    foreach ($wanted in $SupportedPython) {
        foreach ($candidate in $candidates) {
            if ((Test-PythonCandidate $candidate) -eq $wanted) {
                return @{ Exe = $candidate; Version = $wanted }
            }
        }
    }

    return $null
}

function Install-Python {
    <#
        Install Python via winget and return the new interpreter.
        Throws if winget is unavailable or the install does not take.
    #>
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw @"
Python $PythonToInstall is required but not installed, and winget is not
available to install it automatically.

Install Python $PythonToInstall manually from:
    https://www.python.org/downloads/

Tick "Add python.exe to PATH" during installation, then re-run this script.
"@
    }

    Write-Info "Installing Python $PythonToInstall via winget (about 30 MB)..."

    winget install --id $WingetPythonId --exact --source winget `
        --accept-package-agreements --accept-source-agreements `
        --silent --disable-interactivity | Out-Host

    # Do not trust winget's exit code alone - verify we can actually see it.
    $python = Find-Python
    if (-not $python) {
        throw @"
Python was installed but could not be located afterwards.

Close this window, open a new terminal (so PATH refreshes) and re-run
setup.bat. If it still fails, install Python $PythonToInstall manually from
https://www.python.org/downloads/ with "Add python.exe to PATH" ticked.
"@
    }

    return $python
}


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

function Get-GpuVendor {
    <#
        Best-effort GPU vendor detection: NVIDIA, AMD, Intel or Unknown.

        Virtual display adapters (Meta/Oculus streaming monitors, Parsec, RDP)
        are filtered out - they are not real GPUs and would otherwise mask the
        physical card.
    #>
    try {
        $adapters = Get-CimInstance Win32_VideoController -ErrorAction Stop |
            Where-Object { $_.Name -and $_.Name -notmatch 'Virtual|Meta |Parsec|Remote|Basic Display|Citrix' }
    } catch {
        return @{ Vendor = 'Unknown'; Name = 'detection failed' }
    }

    foreach ($vendor in @('NVIDIA', 'AMD', 'Intel')) {
        $match = $adapters | Where-Object {
            $_.Name -match $vendor -or ($vendor -eq 'AMD' -and $_.Name -match 'Radeon')
        } | Select-Object -First 1

        if ($match) { return @{ Vendor = $vendor; Name = $match.Name } }
    }

    $first = $adapters | Select-Object -First 1
    if ($first) { return @{ Vendor = 'Unknown'; Name = $first.Name } }
    return @{ Vendor = 'Unknown'; Name = 'no display adapter found' }
}

function Resolve-Backend {
    <#
        Turn -Backend auto into a concrete choice based on the detected GPU.

        AMD and Intel get Vulkan: it is the backend llama.cpp offers that works
        reliably across the whole range on Windows. (ROCm/HIP builds exist but
        cover far fewer cards and are ten times the download.)
    #>
    param([string]$Requested, [hashtable]$Gpu)

    if ($Requested -ne 'auto') { return $Requested }

    switch ($Gpu.Vendor) {
        'NVIDIA' { return 'cuda' }
        'AMD' { return 'vulkan' }
        'Intel' { return 'vulkan' }
        default { return 'cpu' }
    }
}


# ---------------------------------------------------------------------------
# llama.cpp runtime
# ---------------------------------------------------------------------------

function Get-LlamaRelease {
    <#
        Fetch release metadata from GitHub.

        Args:
            Build: a tag such as 'b10068', or 'latest'.

        Returns the parsed release object.
    #>
    param([string]$Build)

    $uri = if ($Build -eq 'latest') {
        "https://api.github.com/repos/$LlamaRepo/releases/latest"
    } else {
        "https://api.github.com/repos/$LlamaRepo/releases/tags/$Build"
    }

    try {
        return Invoke-RestMethod -Uri $uri -TimeoutSec 60 -Headers @{ 'User-Agent' = 'Athena-Setup' }
    } catch {
        throw @"
Could not reach GitHub to look up the llama.cpp runtime ($Build).

Check your internet connection and try again. If you are behind a proxy or
firewall, you can install the runtime manually - see docs/SETUP.md.

Underlying error: $($_.Exception.Message)
"@
    }
}

function Select-RuntimeAssets {
    <#
        Pick the release assets to download for a backend.

        CUDA needs two archives: the llama.cpp binaries and NVIDIA's CUDA
        runtime DLLs, which are distributed separately.

        Returns an array of asset objects.
    #>
    param($Release, [string]$Backend)

    # Match on the backend segment of the filename, e.g. '-bin-win-vulkan-x64'.
    $patterns = switch ($Backend) {
        'vulkan' { @('bin-win-vulkan-x64\.zip$') }
        'cpu' { @('bin-win-cpu-x64\.zip$') }
        'cuda' { @('bin-win-cuda-\d+\.\d+-x64\.zip$', '^cudart-.*win-cuda-\d+\.\d+-x64\.zip$') }
        default { throw "No runtime mapping for backend '$Backend'." }
    }

    $selected = @()
    foreach ($pattern in $patterns) {
        $asset = $Release.assets |
            Where-Object { $_.name -match $pattern } |
            Sort-Object name |
            Select-Object -First 1

        if (-not $asset) {
            throw @"
The llama.cpp release '$($Release.tag_name)' has no asset matching '$pattern'.

Try a different build:
    setup.bat -LlamaBuild latest
"@
        }
        $selected += $asset
    }

    return $selected
}

function Install-LlamaRuntime {
    <#
        Download and extract the prebuilt llama.cpp runtime.

        Skips the work when a runtime is already present unless -Force.
    #>
    param([string]$Backend)

    $serverExe = Join-Path $RuntimeDir 'llama-server.exe'
    $stampFile = Join-Path $RuntimeDir '.athena-runtime.json'

    if ((Test-Path $serverExe) -and -not $Force) {
        $description = 'unknown build'
        if (Test-Path $stampFile) {
            try {
                $stamp = Get-Content $stampFile -Raw | ConvertFrom-Json
                $description = "$($stamp.backend), $($stamp.build)"
            } catch { }
        }
        Write-Ok "llama.cpp runtime already installed ($description)"
        Write-Info "Re-download with: setup.bat -Force"
        return
    }

    Write-Step "Downloading the llama.cpp runtime ($Backend)"

    $release = Get-LlamaRelease -Build $LlamaBuild
    $assets = Select-RuntimeAssets -Release $release -Backend $Backend

    $totalMb = [math]::Round((($assets | Measure-Object -Property size -Sum).Sum) / 1MB, 1)
    Write-Info "Release $($release.tag_name) - $totalMb MB total"

    if ($Backend -eq 'cuda' -and -not $Yes) {
        Write-Warn "The CUDA runtime is a large download ($totalMb MB)."
        Write-Info "Vulkan is much smaller and also works on NVIDIA: setup.bat -Backend vulkan"
    }

    # Extract into a staging folder first so a failed download cannot leave a
    # half-populated runtime behind.
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) "athena-runtime-$PID"
    if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
    New-Item -ItemType Directory -Path $staging -Force | Out-Null

    try {
        foreach ($asset in $assets) {
            $sizeMb = [math]::Round($asset.size / 1MB, 1)
            Write-Info "Downloading $($asset.name) ($sizeMb MB)..."

            $zipPath = Join-Path $staging $asset.name

            # Invoke-WebRequest's progress bar is extremely slow for large files
            # in Windows PowerShell; suppressing it speeds the download up a lot.
            $previousProgress = $ProgressPreference
            $ProgressPreference = 'SilentlyContinue'
            try {
                Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath -TimeoutSec 1800
            } finally {
                $ProgressPreference = $previousProgress
            }

            Write-Info "Extracting $($asset.name)..."
            Expand-Archive -Path $zipPath -DestinationPath $staging -Force
            Remove-Item $zipPath -Force
        }

        # Some archives nest everything inside a single folder; flatten it so
        # llama-server.exe always lands at the root of the runtime directory.
        $stagedServer = Get-ChildItem $staging -Recurse -Filter 'llama-server.exe' |
            Select-Object -First 1
        if (-not $stagedServer) {
            throw "The downloaded archive did not contain llama-server.exe."
        }
        $sourceDir = $stagedServer.Directory.FullName

        if (Test-Path $RuntimeDir) { Remove-Item $RuntimeDir -Recurse -Force }
        New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
        Copy-Item -Path (Join-Path $sourceDir '*') -Destination $RuntimeDir -Recurse -Force

        # The CUDA runtime DLLs may sit beside, not under, the binaries.
        if ($sourceDir -ne $staging) {
            Get-ChildItem $staging -File -Filter '*.dll' |
                Copy-Item -Destination $RuntimeDir -Force
        }

        @{
            backend = $Backend
            build   = $release.tag_name
            date    = (Get-Date).ToString('s')
        } | ConvertTo-Json | Set-Content $stampFile -Encoding utf8

        Write-Ok "Runtime installed to runtime\llama ($Backend, $($release.tag_name))"
    } finally {
        Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Test-LlamaRuntime {
    <#
        Verify the runtime actually runs and report the devices it can see.
        A failure here is a warning, not an error: the user may simply need a
        driver update, and the rest of the install is still valid.
    #>
    $bench = Join-Path $RuntimeDir 'llama-bench.exe'
    if (-not (Test-Path $bench)) { return }

    # llama-bench writes its device banner to stderr. Under
    # $ErrorActionPreference = 'Stop', merging a native command's stderr turns
    # each line into a terminating error, so relax it just for this call.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $bench --list-devices 2>&1 | Out-String
    } catch {
        Write-Warn "Could not query the runtime for available devices."
        return
    } finally {
        $ErrorActionPreference = $previousPreference
    }

    $devices = ($output -split "`n") |
        Where-Object { $_ -match '^\s+(Vulkan|CUDA|ROCm|SYCL)\d+:' } |
        ForEach-Object { $_.Trim() }

    if ($devices) {
        foreach ($device in $devices) { Write-Ok "GPU available: $device" }
    } else {
        Write-Warn "The runtime reported no GPU devices - Athena will run on the CPU."
        Write-Info "Updating your graphics drivers usually fixes this."
    }
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "  Athena setup" -ForegroundColor White
Write-Host "  ------------" -ForegroundColor DarkGray
Write-Host "  $RepoRoot" -ForegroundColor DarkGray

# --- 1. Python --------------------------------------------------------------

Write-Step "Checking for a supported Python (3.10-3.13)"

$python = Find-Python
if ($python) {
    Write-Ok "Python $($python.Version) at $($python.Exe)"
} else {
    Write-Info "No supported Python found."
    $python = Install-Python
    Write-Ok "Python $($python.Version) at $($python.Exe)"
}

# --- 2. Virtual environment -------------------------------------------------

Write-Step "Preparing the virtual environment (.venv)"

$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'

if (Test-Path $VenvPython) {
    # A venv built against a Python that has since been removed or upgraded
    # fails in confusing ways; rebuild rather than limp along.
    $venvVersion = Test-PythonCandidate $VenvPython
    if ($venvVersion) {
        Write-Ok "Existing virtual environment (Python $venvVersion)"
    } else {
        Write-Warn "Existing .venv is broken - recreating it"
        Remove-Item (Join-Path $RepoRoot '.venv') -Recurse -Force
    }
}

if (-not (Test-Path $VenvPython)) {
    & $python.Exe -m venv (Join-Path $RepoRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
    Write-Ok "Created .venv"
}

Write-Info "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip --quiet | Out-Host

# --- 3. GPU / backend -------------------------------------------------------

Write-Step "Detecting hardware"

$gpu = Get-GpuVendor
Write-Ok "GPU: $($gpu.Name)"

$resolved = Resolve-Backend -Requested $Backend -Gpu $gpu
if ($Backend -eq 'auto') {
    Write-Ok "Selected runtime: $resolved (auto-detected)"
} else {
    Write-Ok "Selected runtime: $resolved (requested)"
}

# --- 4. Python dependencies -------------------------------------------------

Write-Step "Installing Python dependencies"

# llama-cpp-python is deliberately excluded: Athena's default provider uses the
# prebuilt llama.cpp binaries instead, so nothing needs compiling. requirements
# .txt lists it as an optional extra for people who want in-process inference.
$requirementsPath = Join-Path $RepoRoot 'requirements.txt'
$otherLines = Get-Content $requirementsPath | Where-Object { $_ -notmatch '^\s*llama-cpp-python' }

$tempRequirements = Join-Path ([System.IO.Path]::GetTempPath()) "athena-requirements-$PID.txt"
$otherLines | Set-Content $tempRequirements -Encoding utf8

try {
    & $VenvPython -m pip install -r $tempRequirements | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Failed to install dependencies from requirements.txt." }
    Write-Ok "Core dependencies installed"
} finally {
    Remove-Item $tempRequirements -ErrorAction SilentlyContinue
}

if ($Dev) {
    Write-Info "Installing development dependencies..."
    & $VenvPython -m pip install -r (Join-Path $RepoRoot 'requirements-dev.txt') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Failed to install development dependencies." }
    Write-Ok "Development dependencies installed"
}

# --- 5. llama.cpp runtime ---------------------------------------------------

if ($resolved -eq 'none') {
    Write-Step "Skipping the llama.cpp runtime (-Backend none)"
    Write-Info "Athena will need an external provider. See docs/SETUP.md for LM Studio."
} else {
    Install-LlamaRuntime -Backend $resolved
    Test-LlamaRuntime
}

# --- 6. Model folders -------------------------------------------------------

Write-Step "Creating model folders"

# models/ is gitignored in full, so a fresh clone has no model folders at all.
foreach ($folder in @('models\reason', 'models\learning')) {
    $path = Join-Path $RepoRoot $folder
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
    Write-Ok $folder
}

# --- 7. Done ----------------------------------------------------------------

Write-Host ""
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host ""

if ($resolved -eq 'none') {
    Write-Host "  Next: start LM Studio, load a model, and enable its local server." -ForegroundColor White
    Write-Host "  Then set provider to 'lmstudio' - see docs/SETUP.md." -ForegroundColor White
} else {
    Write-Host "  Next: download a GGUF model and put it in:" -ForegroundColor White
    Write-Host "      models\reason\" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Athena will suggest sizes that suit your hardware on first run." -ForegroundColor Gray
}

Write-Host ""
Write-Host "  Start Athena with:  Athena.bat" -ForegroundColor White
Write-Host ""
