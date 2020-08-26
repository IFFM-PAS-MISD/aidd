

cd C:\folder
setlocal enabledelayedexpansion
for %%a in (input*.png) do (
set f=%%a
set f=!f:^(=!
set f=!f:^)=!
ren "%%a" "!f!"
)
