# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AirLLM Chat Windows binary.
Build with:  pyinstaller airllm-chat.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

# Collect airllm package and dependencies
airllm_hiddenimports = collect_submodules('airllm')
airllm_datas = collect_data_files('airllm')

# Additional common hidden imports for ML libraries
hidden_imports = [
    'airllm',
    'torch',
    'numpy',
    'transformers',
    'tokenizers',
    'huggingface_hub',
    'accelerate',
    'safetensors',
    'gguf',
    'tqdm',
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
    'idna',
    'filelock',
    'yaml',
    'packaging',
    'jinja2',
    'markupsafe',
] + airllm_hiddenimports

datas = [
    (os.path.join(PROJECT_ROOT, 'templates', 'index.html'), 'templates'),
    (os.path.join(PROJECT_ROOT, 'static', 'style.css'), 'static'),
    (os.path.join(PROJECT_ROOT, 'static', 'app.js'), 'static'),
    (os.path.join(PROJECT_ROOT, 'VERSION'), '.'),
] + airllm_datas

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'app.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'IPython',
        'notebook',
        'jupyter',
        'PIL',
        'scipy',
        'pandas',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='airllm-chat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for logs (change to False for pure GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add an .ico file path if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='airllm-chat',
)
