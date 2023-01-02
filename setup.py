from setuptools import setup
setup(
    install_requires=[
        'h5py>3.6',
        'matplotlib>3.5',
        'numpy>1.23.2',
        'pillow',
        'pandas>1.4.1',
        'seaborn',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'PyQt5',
        'PyOpenGL',
        'pyqtgraph'],
    script=['examples/oda-preview']
    )
