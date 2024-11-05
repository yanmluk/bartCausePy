import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

def init_R():
    base = rpackages.importr('base')
    utils = rpackages.importr('utils')

    # extra packages to install
    packnames = ('bartCause', 'dplyr')
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        print("r packages need to install:", names_to_install)
        utils.install_packages(StrVector(names_to_install))