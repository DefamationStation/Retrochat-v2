[Setup]
AppName=RetroChat
AppVersion=0.1
DefaultDirName={userappdata}\.retrochat
DefaultGroupName=RetroChat
UninstallDisplayIcon={app}\retrochat.exe
Compression=lzma2
SolidCompression=yes
OutputDir=userdocs:Inno Setup Examples Output

[Files]
Source: "C:\Users\frenz\Downloads\LLMs\Retrochat-backend-and-frontend\dist\retrochat.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\RetroChat"; Filename: "{app}\retrochat.exe"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER,
    'Environment',
    'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

[Run]
Filename: "{cmd}"; Parameters: "/C mklink ""{userdocs}\rchat.bat"" ""{app}\retrochat.exe"""; Flags: runhidden

[UninstallDelete]
Type: files; Name: "{userdocs}\rchat.bat"