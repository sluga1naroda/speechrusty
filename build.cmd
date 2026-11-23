@echo off
rem Windows build wrapper. Usage: build.cmd <cargo args>
rem   build.cmd build --release -p sr-app -p sr-mt-sidecar
rem   build.cmd test --workspace
rem
rem Recent MSBuild breaks ggml's vulkan-shaders-gen ExternalProject, so whisper.cpp / llama.cpp
rem are built with NMake inside the MSVC developer environment (located via vswhere).
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere.exe not found: install Visual Studio with "Desktop development with C++". 1>&2
  exit /b 1
)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
  echo No Visual Studio installation with the C++ x64 toolchain found. 1>&2
  exit /b 1
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set CMAKE_GENERATOR=NMake Makefiles
cargo %*
