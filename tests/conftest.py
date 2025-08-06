from __future__ import annotations

import os
import sys

# Prevent oversubscription when using VULCAN
os.environ["OMP_NUM_THREADS"] = "2" # noqa

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
