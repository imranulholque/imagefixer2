"""Windows extras: start with Windows, and the right-click > Send to entry."""
import os
import subprocess
import sys
import threading

IS_WIN = sys.platform == "win32"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "ImageFixer"


def exe_command():
    """List that starts this program (the .exe when packed, python + script when run from source)."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, os.path.abspath(sys.argv[0])]


def _quote(parts):
    return " ".join('"%s"' % p for p in parts)


def is_autostart():
    if not IS_WIN:
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ) as k:
            winreg.QueryValueEx(k, RUN_NAME)
            return True
    except OSError:
        return False


def set_autostart(on):
    if not IS_WIN:
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            if on:
                winreg.SetValueEx(k, RUN_NAME, 0, winreg.REG_SZ, _quote(exe_command()) + " --tray")
            else:
                try:
                    winreg.DeleteValue(k, RUN_NAME)
                except FileNotFoundError:
                    pass
    except OSError:
        pass


def sendto_link():
    return os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "SendTo", "Image Fixer.lnk")


def is_sendto():
    return IS_WIN and os.path.exists(sendto_link())


def _ps_quote(s):
    return "'" + s.replace("'", "''") + "'"


def _make_link():
    cmd = exe_command()
    target = cmd[0]
    args = _quote(cmd[1:]) if len(cmd) > 1 else ""
    script = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut(%s);"
              "$s.TargetPath=%s;$s.Arguments=%s;$s.WorkingDirectory=%s;$s.IconLocation=%s;$s.Save()") % (
        _ps_quote(sendto_link()), _ps_quote(target), _ps_quote(args),
        _ps_quote(os.path.dirname(target)), _ps_quote(target + ",0"))
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-Command", script], creationflags=0x08000000, timeout=20,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except Exception:
        pass


def set_sendto(on):
    """Adds / removes the 'Image Fixer' entry in Explorer's right-click > Send to menu (runs in background)."""
    if not IS_WIN:
        return

    def work():
        if on:
            os.makedirs(os.path.dirname(sendto_link()), exist_ok=True)
            _make_link()
        else:
            try:
                os.remove(sendto_link())
            except OSError:
                pass
    threading.Thread(target=work, daemon=True).start()
