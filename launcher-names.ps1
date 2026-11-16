# Shared by the launchers (Align LRC.bat, Review LRC.bat).
#
# cmd.exe expands %NAME% in a command line given with its /c switch before the batch runs,
# so a dropped file called "a%TEMP%b.flac" reaches %* as "a<temp path>b.flac". The
# unexpanded text is still in the command line of the parent cmd.exe: take a name
# from there only when expanding it gives exactly the name that was received.
function Restore-DroppedNames {
    param(
        [string[]] $Received,
        # Full path of the launcher batch file: its name (long and 8.3 form) marks where the dropped names start.
        [string] $LauncherPath
    )
    try {
        $parentId = (Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$PID" -ErrorAction Stop).ParentProcessId
        $parentLine = [string](Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$parentId" -ErrorAction Stop).CommandLine
        $names = @([System.IO.Path]::GetFileName($LauncherPath))
        try {
            $shortName = (New-Object -ComObject Scripting.FileSystemObject).GetFile($LauncherPath).ShortName
            if ($shortName) { $names += $shortName }
        }
        catch { }
        $batchMarker = -1
        $markerLength = 0
        foreach ($name in $names) {
            $found = $parentLine.IndexOf($name, [System.StringComparison]::OrdinalIgnoreCase)
            if ($found -ge 0) { $batchMarker = $found; $markerLength = $name.Length; break }
        }
        if ($batchMarker -lt 0) { return $Received }
        $originalTokens = @()
        foreach ($match in [regex]::Matches($parentLine.Substring($batchMarker + $markerLength).TrimStart('"'), '"([^"]*)"|(\S+)')) {
            if ($match.Groups[1].Success) { $originalTokens += $match.Groups[1].Value } else { $originalTokens += $match.Groups[2].Value }
        }
        $recoveredPaths = @()
        foreach ($receivedPath in $Received) {
            $recovered = $receivedPath
            foreach ($token in $originalTokens) {
                if (($token -ne $receivedPath) -and
                    [string]::Equals([System.Environment]::ExpandEnvironmentVariables($token), $receivedPath, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $recovered = $token
                    break
                }
            }
            $recoveredPaths += $recovered
        }
        return $recoveredPaths
    }
    catch {
        Write-Warning ("Could not recover the dropped file names from the launcher: {0}" -f $_.Exception.Message)
        return $Received
    }
}
