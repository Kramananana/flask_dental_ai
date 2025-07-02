# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 包含所有HTML模板和静态文件
        ('templates', 'templates'),
        ('static', 'static'),
        # 包含AI模型、上传、聊天记录等所有程序需要读写的文件夹
        ('runs', 'runs'),
        ('uploads', 'uploads'),
        ('chat_logs', 'chat_logs'),
        ('patients_data', 'patients_data'),
        # 【重要】确保您的数据库文件名正确
        # 如果您希望打包时包含一个初始数据库，请取消这行的注释
        # ('dental_app.db', '.') 
    ],
    hiddenimports=[
        # 强制包含gevent及其所有子模块，解决异步模式的错误
        *collect_submodules('gevent'),
        'engineio.async_drivers.gevent',
        'sqlalchemy.sql.default_comparator'
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
    name='AI牙科助手',
    debug=False,
    boot_loader_options=None,
    strip=False,
    upx=True,
    # 如果您想在运行时看到黑色命令行窗口以进行调试，可以暂时设为True
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None # 您可以在这里指定一个.ico图标文件的路径
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI牙科助手'
)