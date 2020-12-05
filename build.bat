@echo off
call run_constants.bat
set file_arg=%1
for /F "delims=" %%i in ("%file_arg%") do set filename="%%~ni"

cl %1 /Zi /GL /GR- /EHsc /nologo /Bt /std:c++latest /fp:strict /diagnostics:caret /I . /I src\ /Fo:%BIN_DIR%\ /DUNICODE /link /OUT:%BIN_DIR%/%filename%.exe

if %ERRORLEVEL% EQU 0 (
  if "%~2" == "-r" (
    %BIN_DIR%\%filename%.exe
  )
)
