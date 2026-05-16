# pylint: disable=wrong-import-position, wrong-import-order, invalid-name
"""
Invoke build script.
Show all tasks with::
    invoke -l
.. seealso::
    * http://pyinvoke.org
    * https://github.com/pyinvoke/invoke
"""
###############################################################################
# Catch exceptions and go into ipython/ipdb
# import sys

# from IPython.core.debugger import Tracer  # noqa
# from IPython.core import ultratb

# sys.excepthook = ultratb.FormattedTB(
#     mode="Verbose", color_scheme="Linux", call_pdb=True, ostream=sys.__stdout__
# )
###############################################################################

from invoke import Collection

from . import ci, local

ns = Collection()
ns.add_collection(local)
ns.add_collection(ci)
