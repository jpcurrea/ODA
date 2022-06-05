from setuptools import setup
setup(
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'pillow',
        'pandas',
        'seaborn',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'PyQt5',
        'PyOpenGL',
        'pyqtgraph'],
    script=['./examples/oda-preview']
    )
