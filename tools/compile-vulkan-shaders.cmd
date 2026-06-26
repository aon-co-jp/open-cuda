@echo off
setlocal
set ROOT=%~dp0..
set SHADER_DIR=%ROOT%\examples\vector_add_vulkan_real\shaders
set SRC=%SHADER_DIR%\vector_add.comp
set OUT=%SHADER_DIR%\vector_add.spv

where glslc >nul 2>nul
if errorlevel 1 (
  echo ERROR: glslc が見つかりません。Vulkan SDK をインストールして、C:\VulkanSDK\^<version^>\Bin を PATH に追加して下さい。
  exit /b 1
)

glslc "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo OK: compiled %SRC% -^> %OUT%
