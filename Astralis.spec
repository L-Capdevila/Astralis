# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Script intelligent : on cherche les fichiers qui existent vraiment pour ne pas faire paniquer PyInstaller
fichiers_a_inclure = [
    ('settings.json', '.'),
    ('moteur_astralis.py', '.'),
    ('config.py', '.'), 
    ('mode_d_emploi.txt', '.'),
    ('core', 'core'),
]

# Si des logos existent, on les ajoute. Sinon, on ignore sans bloquer.
for nom_logo in ['astralis_logo.png', 'astralis_logo.jpg', 'logo.png', 'icon.png', 'icon.ico']:
    if os.path.exists(nom_logo):
        fichiers_a_inclure.append((nom_logo, '.'))

a = Analysis(
    ['Astralis.py'],
    pathex=[],
    binaries=[],
    datas=fichiers_a_inclure,
    hiddenimports=[
        'pandas', 'numpy', 'numba', 'scipy', 'tqdm', 'pyarrow',
        'matplotlib', 'PyQt5', 'flask', 'flask_cors'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='Astralis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Cache la console noire de Windows en arrière-plan
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Astralis',
)