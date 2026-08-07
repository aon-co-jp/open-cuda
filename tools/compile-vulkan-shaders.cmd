@echo off
setlocal
set ROOT=%~dp0..

where glslc >nul 2>nul
if errorlevel 1 (
  echo ERROR: glslc が見つかりません。Vulkan SDK をインストールして、C:\VulkanSDK\^<version^>\Bin を PATH に追加して下さい。
  exit /b 1
)

echo glslc version:
glslc --version

set SHADER_DIR=%ROOT%\examples\vector_add_vulkan_real\shaders
set SRC=%SHADER_DIR%\vector_add.comp
set OUT=%SHADER_DIR%\vector_add.spv
glslc "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo OK: compiled %SRC% -^> %OUT%

set SHADER_DIR=%ROOT%\examples\matmul_vulkan_real\shaders
set SRC=%SHADER_DIR%\matmul.comp
set OUT=%SHADER_DIR%\matmul.spv
glslc "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo OK: compiled %SRC% -^> %OUT%

set SHADER_DIR=%ROOT%\examples\raid6_xor_parity_vulkan_real\shaders
set SRC=%SHADER_DIR%\raid6_xor_parity.comp
set OUT=%SHADER_DIR%\raid6_xor_parity.spv
glslc "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo OK: compiled %SRC% -^> %OUT%

set SHADER_DIR=%ROOT%\examples\raid6_q_parity_vulkan_real\shaders
set SRC=%SHADER_DIR%\raid6_q_parity.comp
set OUT=%SHADER_DIR%\raid6_q_parity.spv
glslc "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo OK: compiled %SRC% -^> %OUT%

set SHADER_DIR=%ROOT%\examples\softmax_vulkan_real\shaders
set SRC=%SHADER_DIR%\softmax.comp
set OUT=%SHADER_DIR%\softmax.spv
glslc "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo OK: compiled %SRC% -^> %OUT%

set SHADER_DIR=%ROOT%\examples\flash_attention_vulkan_real\shaders
set SRC=%SHADER_DIR%\flash_attention.comp
set OUT=%SHADER_DIR%\flash_attention.spv
glslc "%SRC%" -o "%OUT%"
if errorlevel 1 exit /b %errorlevel%
echo OK: compiled %SRC% -^> %OUT%
