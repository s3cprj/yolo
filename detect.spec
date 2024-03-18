# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
import ultralytics
import sys
from os import path

ultralytics_path = path.join(sys.prefix, 'Lib', 'site-packages', 'ultralytics', 'cfg', 'default.yaml')
ultra_files = collect_data_files('ultralytics')

a = Analysis(
    ['detect.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models'),('requirements.txt', '.')] + ultra_files,
    hiddenimports=['os', 'sys', 'torch', 'git', 'numpy', 'matplotlib', 'cv2', 'PIL', 'psutil', 'yaml', 'requests', 'scipy', 'torchvision', 'tqdm', 'ultralytics', 'pandas', 'seaborn', 'setuptools'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
a.datas += [('ultralytics/cfg/default.yaml', ultralytics_path, 'DATA')]
a.datas += [('utils/general.pyc', 'utils/general.py', 'DATA')]
a.datas += [('exp30/weights/best.pt', 'exp30/weights/best.pt', 'DATA')]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='detect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)