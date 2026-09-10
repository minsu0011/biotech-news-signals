param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class V221NativeLauncher {
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] public struct STARTUPINFO { public Int32 cb; public string lpReserved; public string lpDesktop; public string lpTitle; public Int32 dwX; public Int32 dwY; public Int32 dwXSize; public Int32 dwYSize; public Int32 dwXCountChars; public Int32 dwYCountChars; public Int32 dwFillAttribute; public Int32 dwFlags; public Int16 wShowWindow; public Int16 cbReserved2; public IntPtr lpReserved2; public IntPtr hStdInput; public IntPtr hStdOutput; public IntPtr hStdError; }
  [StructLayout(LayoutKind.Sequential)] public struct PROCESS_INFORMATION { public IntPtr hProcess; public IntPtr hThread; public Int32 dwProcessId; public Int32 dwThreadId; }
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] public static extern bool CreateProcess(string app, StringBuilder command, IntPtr pa, IntPtr ta, bool inherit, UInt32 flags, IntPtr environment, string cwd, ref STARTUPINFO si, out PROCESS_INFORMATION pi);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetProcessAffinityMask(IntPtr process, UIntPtr mask);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GetProcessAffinityMask(IntPtr process, out UIntPtr processMask, out UIntPtr systemMask);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern UInt32 ResumeThread(IntPtr thread);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern UInt32 WaitForSingleObject(IntPtr handle, UInt32 milliseconds);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GetExitCodeProcess(IntPtr process, out UInt32 code);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool CloseHandle(IntPtr handle);
}
'@
$python = 'C:\Users\minsu\anaconda3\python.exe'
$quoted = foreach ($arg in $PythonArgs) { '"' + ($arg -replace '"', '\"') + '"' }
$command = [Text.StringBuilder]::new(('"' + $python + '" ' + ($quoted -join ' ')))
$startup = [V221NativeLauncher+STARTUPINFO]::new(); $startup.cb = [Runtime.InteropServices.Marshal]::SizeOf($startup)
$process = [V221NativeLauncher+PROCESS_INFORMATION]::new()
$created = [V221NativeLauncher]::CreateProcess($python, $command, [IntPtr]::Zero, [IntPtr]::Zero, $true, 0x4, [IntPtr]::Zero, (Get-Location).Path, [ref]$startup, [ref]$process)
if (-not $created) { throw "CreateProcess failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())" }
try {
  if (-not [V221NativeLauncher]::SetProcessAffinityMask($process.hProcess, [UIntPtr]([UInt64]3221225472))) { throw "SetProcessAffinityMask failed" }
  [UIntPtr]$observedMask = [UIntPtr]::Zero; [UIntPtr]$systemMask = [UIntPtr]::Zero
  if (-not [V221NativeLauncher]::GetProcessAffinityMask($process.hProcess, [ref]$observedMask, [ref]$systemMask)) { throw "GetProcessAffinityMask failed" }
  if ($observedMask.ToUInt64() -ne [UInt64]3221225472) { throw "pre-resume affinity mismatch" }
  Write-Output ("V221_EXTERNAL_LAUNCH pid={0} pre_resume_mask=0xC0000000 cpus=30,31" -f $process.dwProcessId)
  if ([V221NativeLauncher]::ResumeThread($process.hThread) -eq 0xFFFFFFFF) { throw "ResumeThread failed" }
  [void][V221NativeLauncher]::WaitForSingleObject($process.hProcess, [UInt32]::MaxValue)
  [UInt32]$exitCode = 1; if (-not [V221NativeLauncher]::GetExitCodeProcess($process.hProcess, [ref]$exitCode)) { throw "GetExitCodeProcess failed" }
  exit [int]$exitCode
} finally { [void][V221NativeLauncher]::CloseHandle($process.hThread); [void][V221NativeLauncher]::CloseHandle($process.hProcess) }
