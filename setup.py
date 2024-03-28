"""Setup file for ml_for_national_blend."""

from setuptools import setup

PACKAGE_NAMES = [
    'ml_for_national_blend', 'ml_for_national_blend.io',
    'ml_for_national_blend.utils', 'ml_for_national_blend.machine_learning',
    'ml_for_national_blend.plotting', 'ml_for_national_blend.scripts'
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data-mining', 'weather', 'meteorology', 'wildfire', 'forest fire',
    'fire weather', 'bias correction', 'post-processing',
    'uncertainty quantification', 'MOS', 'model-output statistics'
]
SHORT_DESCRIPTION = (
    'Machine learning for post-processing National Blend of [Weather] Models '
    '(NBM).'
)
LONG_DESCRIPTION = (
    'ml_for_national_blend is an end-to-end library that uses machine learning '
    '(ML) to post-process the National Blend of [Weather] Models (NBM).  Think '
    'of this as gridded model-output statistics (MOS) with machine learning '
    'and uncertainty quantification, i.e., the output for every weather '
    'variable is a grid with an ensemble of forecasts at every grid point.'
)
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

PACKAGE_REQUIREMENTS = [
    'numpy',
    'xarray',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'descartes',
    'geopy',
    'metpy'
]

if __name__ == '__main__':
    setup(
        name='ml_for_national_blend',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ralager@colostate.edu',
        url='https://github.com/thunderhoser/ml_for_national_blend',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
