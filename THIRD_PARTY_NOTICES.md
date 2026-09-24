# Third-party notices

`model/s4.py` contains code derived from Albert Gu and contributors'
state-spaces/S4 implementation. The upstream source
is https://github.com/state-spaces/s4 and its Apache-2.0 license is reproduced in
`licenses/S4-LICENSE`. The project modifies the S4 wrapper and includes a native
PyTorch Cauchy fallback.

`layers/`, `utils/masking.py`, and portions of the model/training utilities derive
from THUML/iTransformer (Copyright (c) 2022 THUML @ Tsinghua University),
https://github.com/thuml/iTransformer. The MIT license and notice are reproduced
in `licenses/iTransformer-LICENSE`.

All other contributions: Copyright 2026 S4-iTNet contributors, Apache-2.0.
The license does not grant rights to TUSZ EEG data. Obtain that corpus under its
own terms. The included smoke signals are generated mathematically.
