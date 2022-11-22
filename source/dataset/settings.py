from datetime import datetime
import os

import path

CURRENT_ROOT = path.Path(os.path.dirname(__file__)).abspath()
DATA_ROOT = CURRENT_ROOT.joinpath("data")
