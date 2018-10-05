rmdir /S /Q hoppinworld-export-windows
mkdir hoppinworld-export-windows
xcopy .\target\release\hoppinworld.exe hoppinworld-export-windows\
xcopy .\assets hoppinworld-export-windows\assets\ /E
xcopy .\maps hoppinworld-export-windows\maps\ /E
"C:\Program Files\7-Zip\7z.exe" a -r hoppinworld-windows.zip -w hoppinworld-export-windows -mem=AES256

pause