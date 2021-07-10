$UIDir = "..\ui_files"
$PYDir = ".\windows\design"
$TS_Dir = "..\languages"
$QM_Dir = ".\resources\translations"
$PY_Dir = ".\windows"
$Languages = "en", "de", "ja", "fil", "ru", "tr"
Set-Location $(Split-Path -Path $MyInvocation.MyCommand.Path)
# Get Files
$FileNames = New-Object Collections.Generic.List[String]
Get-ChildItem $UIDir -Filter *.ui | 
Foreach-Object {
    $FileNames.Add($_.FullName)
}
$FileTimes = [System.Collections.ArrayList]@()
Foreach ($FileName in $FileNames) {
    $FileTimes.Add($(Get-Date))
}
$ChangedFiles = [System.Collections.ArrayList]@()

# autopep8 --in-place <filename>
# pyside6-uic <filebasename>.ui > <filebasename>.py
# lupdate -no-obsolete -silent <filebasename>.ui -ts ../../languages/$(<filebasename>)_de.ts
# endless loop
for () {
    $i = 0
    Foreach ($FileName in $FileNames) {
        $file = Get-Item $FileName
        if ($FileTimes[$i] -ne $file.LastWriteTime) {
            $basename = $($file.basename)
            Write-Output "Updating $basename"
            Write-Output "`tGenerating file"
            cmd.exe /c "pyside2-uic `"$UIDir\$basename.ui`" > `"$PYDir\$($basename)_ui.py`""
            Write-Output "`tFormatting file"
            cmd.exe /c "autopep8 --in-place $PYDir\$($basename)_ui.py"
            $ChangedFiles.Add($basename)
        }
        $FileTimes[$i] = $file.LastWriteTime
        $i += 1
    }
    if ($ChangedFiles.Count -gt 0) {
        Write-Output "Generating Language and QML Files"
        
        $files = ''
        Foreach ($FileName in $FileNames) {
            $basename = $(Get-Item $FileName).basename
            $files += "$UIDir\$basename.ui "
            $files += "$PY_Dir\$basename.py "
        }

        Foreach ($language in $Languages) {
            cmd.exe /c "pyside2-lupdate -noobsolete $files -ts $TS_Dir/$language.qt.ts"
            New-Item -ItemType Directory -Force -Path $QM_Dir/$language | Out-Null
            New-Item -ItemType Directory -Force -Path $QM_Dir/$language/infos | Out-Null
            if ($language -ne "en") {
                Copy-Item -Path "$QM_Dir/en/infos/*" -Destination "$QM_Dir/$language/infos" -PassThru | Out-Null
            }
            cmd.exe /c "lrelease -silent -removeidentical $TS_Dir/$language.qt.ts -qm $QM_Dir/$language/$language.qm"
        }
        $ChangedFiles.Clear()
        Write-Output "Done!"
    }
    Start-Sleep 3
}
Exit