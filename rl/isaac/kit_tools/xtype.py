"""X-level input/screenshot for the Isaac Xvfb display :1 (run with kit python, PYTHONPATH=/workspace/pylib).

    xtype.py click X Y | dclick X Y | type TEXT | key ctrl+Return | shot OUT.ppm [X0 Y0 X1 Y1]
Several commands may be chained with ';;' as separate argv items.
"""
import sys
import time

from Xlib import X, XK, display
from Xlib.ext import xtest

d = display.Display(":1")
root = d.screen().root

SHIFTED = {c: b for b, c in zip("`1234567890-=[]\\;',./", "~!@#$%^&*()_+{}|:\"<>?")}
NAMES = {" ": "space", "\n": "Return", "\t": "Tab", "(": "parenleft", ")": "parenright", "[": "bracketleft",
         "]": "bracketright", "{": "braceleft", "}": "braceright", "'": "apostrophe", '"': "quotedbl",
         ".": "period", ",": "comma", "/": "slash", "\\": "backslash", ";": "semicolon", ":": "colon",
         "=": "equal", "-": "minus", "_": "underscore", "+": "plus", "*": "asterisk", "#": "numbersign",
         "!": "exclam", "@": "at", "$": "dollar", "%": "percent", "^": "asciicircum", "&": "ampersand",
         "<": "less", ">": "greater", "?": "question", "|": "bar", "~": "asciitilde", "`": "grave"}
MODS = {"ctrl": "Control_L", "shift": "Shift_L", "alt": "Alt_L", "super": "Super_L"}


def kc(name):
    code = d.keysym_to_keycode(XK.string_to_keysym(name))
    if not code:
        raise SystemExit(f"no keycode for {name!r}")
    return code


def press(code, down):
    xtest.fake_input(d, X.KeyPress if down else X.KeyRelease, code)
    d.sync()


def key(combo):
    parts = combo.split("+")
    mods, k = [MODS.get(p, p) for p in parts[:-1]], parts[-1]
    for m in mods:
        press(kc(m), True)
    time.sleep(0.05)
    press(kc(k), True)
    time.sleep(0.08)
    press(kc(k), False)
    time.sleep(0.05)
    for m in reversed(mods):
        press(kc(m), False)
    time.sleep(0.03)


def type_text(s):
    for ch in s:
        shift = ch.isupper() or ch in SHIFTED
        base = ch.lower() if ch.isalpha() else SHIFTED.get(ch, ch)
        name = NAMES.get(base, base)
        if shift:
            press(kc("Shift_L"), True)
        press(kc(name), True)
        time.sleep(0.02)
        press(kc(name), False)
        if shift:
            press(kc("Shift_L"), False)
        time.sleep(0.01)


def click(x, y, n=1):
    xtest.fake_input(d, X.MotionNotify, x=int(x), y=int(y))
    d.sync()
    time.sleep(0.05)
    for _ in range(n):
        xtest.fake_input(d, X.ButtonPress, 1)
        d.sync()
        time.sleep(0.08)
        xtest.fake_input(d, X.ButtonRelease, 1)
        d.sync()
        time.sleep(0.08)


def shot(out, x0=0, y0=0, x1=1920, y1=1080):
    w, h = x1 - x0, y1 - y0
    img = root.get_image(x0, y0, w, h, X.ZPixmap, 0xFFFFFFFF)
    data = img.data
    rgb = bytearray(w * h * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = data[2::4], data[1::4], data[0::4]
    with open(out, "wb") as f:
        f.write(b"P6 %d %d 255\n" % (w, h))
        f.write(bytes(rgb))


def run(cmd):
    op, args = cmd[0], cmd[1:]
    if op == "click":
        click(*map(int, args))
    elif op == "dclick":
        click(*map(int, args), n=2)
    elif op == "type":
        type_text(" ".join(args))
    elif op == "key":
        key(args[0])
    elif op == "sleep":
        time.sleep(float(args[0]))
    elif op == "shot":
        shot(args[0], *map(int, args[1:]))
    else:
        raise SystemExit(f"unknown op {op}")


cmd = []
for a in sys.argv[1:] + [";;"]:
    if a == ";;":
        if cmd:
            run(cmd)
        cmd = []
    else:
        cmd.append(a)
