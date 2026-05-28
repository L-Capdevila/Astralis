# -*- mode: python ; coding: utf-8 -*-
import os
import shutil

block_cipher = None
_spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Fichiers embarqués dans _internal/ (bundle PyInstaller)
fichiers_a_inclure = [
    ('settings.json', '.'),
    ('moteur_astralis.py', '.'),
    ('dashboard_orbite.py', '.'),
    ('splash.py', '.'),
    ('config.py', '.'),
    ('mode_d_emploi.txt', '.'),
    ('core', 'core'),
]

# Assets optionnels (logos, viewer web)
for nom in ['astralis_logo.png', 'astralis_logo.jpg', 'logo.png', 'icon.png', 'icon.ico']:
    if os.path.exists(os.path.join(_spec_dir, nom)):
        fichiers_a_inclure.append((nom, '.'))
if os.path.exists(os.path.join(_spec_dir, 'index.html')):
    fichiers_a_inclure.append(('index.html', '.'))

a = Analysis(
    ['Astralis.py'],
    pathex=[_spec_dir],
    binaries=[],
    datas=fichiers_a_inclure,
    hiddenimports=[
        'pandas', 'numpy', 'numba', 'tqdm', 'pyarrow',
        'matplotlib', 'PyQt5', 'flask', 'flask_cors',
        'splash', 'dashboard_orbite', 'moteur_astralis', 'config',
        'core.state', 'core.forces', 'core.integrator', 'core.monitor',
        'core.body_init', 'core.body_positions', 'core.celestial_velocities',
        'core.metrics_sidecar', 'core.metrics_core', 'core.periods',
        'core.realtime_viewer',
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
    console=False,
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

# Copie à côté de Astralis.exe (Program Files) : chemins attendus par le dashboard
_dist = os.path.join(_spec_dir, 'dist', 'Astralis')
for _name in (
    'moteur_astralis.py',
    'dashboard_orbite.py',
    'settings.json',
    'mode_d_emploi.txt',
):
    _src = os.path.join(_spec_dir, _name)
    _dst = os.path.join(_dist, _name)
    if os.path.isfile(_src):
        shutil.copy2(_src, _dst)
