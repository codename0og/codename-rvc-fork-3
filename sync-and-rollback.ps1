# ----------------------------------------
# Interactive Sync & Rollback Script
# ----------------------------------------

# Set working directory to script location
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Define paths
$TempPath    = Join-Path $ScriptDir "fork3-temp"
$RollbackDir = Join-Path $ScriptDir "rollback_cache"
$ThisScript  = $MyInvocation.MyCommand.Name

# Ensure rollback_cache exists
if (!(Test-Path $RollbackDir)) {
    New-Item -ItemType Directory -Path $RollbackDir | Out-Null
}

function Show-Menu {
    Write-Host ""
    Write-Host "Select an option:"
    Write-Host "  1. Sync with repo"
    Write-Host "  2. Rollback"
    $choice = Read-Host "Enter choice (1 or 2)"
    return $choice
}

function Backup-Files {
    param (
        [string]$Name = ""
    )
    # Create timestamped (and optionally named) backup folder
    $timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
    $sanitized  = $Name -replace '[^\w\-]', '_'        # sanitize name
    if ($sanitized) {
        $folderName = "${timestamp}_$sanitized"
    } else {
        $folderName = $timestamp
    }
    $backupPath = Join-Path $RollbackDir $folderName
    New-Item -ItemType Directory -Path $backupPath | Out-Null

    # Copy all files except this script and the rollback_cache folder
    Get-ChildItem -Path $ScriptDir -Recurse -File |
        Where-Object {
            $_.FullName -notlike "$RollbackDir*" -and
            $_.Name       -ne $ThisScript
        } |
        ForEach-Object {
            $relativePath  = $_.FullName.Substring($ScriptDir.Length).TrimStart('\')
            $destPath      = Join-Path $backupPath $relativePath
            $destDir       = Split-Path $destPath -Parent
            if (!(Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir | Out-Null
            }
            Copy-Item -Path $_.FullName -Destination $destPath -Force
        }

    Write-Host " Backup created: $folderName"
}

function Sync-With-Repo {
    Write-Host ""
    Write-Host " Syncing with repo..."

    # Ask for optional snapshot name
    $customName = Read-Host "Enter a name for this backup (optional)"
    Backup-Files -Name $customName

    # Remove old temp folder
    Remove-Item -Recurse -Force $TempPath -ErrorAction Ignore

    # Clone fresh copy
    git clone https://github.com/codename0og/codename-rvc-fork-3.git $TempPath

    # Copy files from temp (excluding this script)
    Get-ChildItem -Path $TempPath -Recurse -File |
        ForEach-Object {
            $relativePath = $_.FullName.Substring($TempPath.Length).TrimStart('\')
            if ($relativePath -ne $ThisScript) {
                $destinationPath = Join-Path $ScriptDir $relativePath
                $destinationDir  = Split-Path $destinationPath -Parent
                if (!(Test-Path $destinationDir)) {
                    New-Item -ItemType Directory -Path $destinationDir | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $destinationPath -Force
            }
        }

    # Cleanup
    Remove-Item -Recurse -Force $TempPath -ErrorAction Ignore

    Write-Host " Sync complete from codename-rvc-fork-3."
}

function Rollback {
    Write-Host ""
    # Get snapshots in descending (newest-first) order
    $snapshots = Get-ChildItem -Directory $RollbackDir | Sort-Object Name -Descending

    if ($snapshots.Count -eq 0) {
        Write-Host " No rollback snapshots available."
        return
    }

    # List available snapshots
    Write-Host "Available rollback snapshots:"
    for ($i = 0; $i -lt $snapshots.Count; $i++) {
        $index = $i + 1
        Write-Host "  $index. $($snapshots[$i].Name)"
    }

    $sel = Read-Host "Enter snapshot number to restore"
    if ($sel -match '^\d+$' -and $sel -ge 1 -and $sel -le $snapshots.Count) {
        $chosen = $snapshots[$sel - 1]
        Write-Host "Restoring snapshot: $($chosen.Name)"

        # Restore files from snapshot
        Get-ChildItem -Path $chosen.FullName -Recurse -File |
            ForEach-Object {
                $relativePath    = $_.FullName.Substring($chosen.FullName.Length).TrimStart('\')
                $destinationPath = Join-Path $ScriptDir $relativePath
                $destinationDir  = Split-Path $destinationPath -Parent
                if (!(Test-Path $destinationDir)) {
                    New-Item -ItemType Directory -Path $destinationDir | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $destinationPath -Force
            }

        Write-Host " Rollback complete."
    }
    else {
        Write-Host " Invalid selection."
    }
}

# ---- Main Execution ----
$choice = Show-Menu
switch ($choice) {
    "1" { Sync-With-Repo }
    "2" { Rollback }
    default { Write-Host " Invalid choice. Exiting." }
}