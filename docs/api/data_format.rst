.. _data-format:

Data Format Specification
=========================

This document specifies the CSV and MAT file formats supported by RingDownAnalysis for ring-down measurement data. The package expects **Moku:Lab Phasemeter** export format.

Supported Formats
-----------------

+----------+-------------+--------------------------------------+
| Format   | Extension   | Loader                               |
+==========+=============+======================================+
| CSV      | ``.csv``    | ``RingDownDataLoader.load_csv()``    |
+----------+-------------+--------------------------------------+
| MATLAB   | ``.mat``    | ``RingDownDataLoader.load_mat()``    |
+----------+-------------+--------------------------------------+

Use ``RingDownDataLoader.load()`` to auto-detect format from file extension.

CSV Format
----------

Column Layout
~~~~~~~~~~~~~

.. list-table:: CSV columns
   :header-rows: 1
   :widths: 15 10 10 12 35

   * - Column Index
     - Name
     - Type
     - Units
     - Description
   * - 0
     - time
     - float
     - seconds
     - Timestamp
   * - 1
     - col1
     - int
     - —
     - Unused (typically 0)
   * - 2
     - col2
     - int
     - —
     - Unused (typically 0)
   * - 3
     - phase
     - float
     - cycles
     - Phase in cycles (primary data)

**Minimum required columns**: 4. The loader reads only columns 0 (time) and 3 (phase).

Structure
~~~~~~~~~

- **Comments**: Lines starting with ``%`` are skipped.
- **Header**: No header row; first non-comment line is data.
- **Delimiter**: Comma (`,`).
- **Encoding**: UTF-8 or ASCII.

Example
~~~~~~~

.. code-block:: text

   % Moku:Lab Phasemeter - time,col1,col2,phase
   0.000000,0,0,0.100000
   0.001000,0,0,0.099618
   0.002000,0,0,0.098566

Post-Processing
~~~~~~~~~~~~~~~

- **Time**: Shifted to start at 0 (i.e., ``t_out = t_raw - t_raw[0]``).
- **Phase**: Detrended with ``scipy.signal.detrend(..., type="constant")``.

Validation Rules
~~~~~~~~~~~~~~~~

- At least one valid data line (non-comment, 4+ columns, numeric first column).
- First column must be parseable as float.
- Raises ``ValueError`` if: empty file, comments only, fewer than 4 columns, or malformed numeric data.

MAT Format
----------

Structure
~~~~~~~~~

MAT files must contain a variable named ``moku`` with a nested ``data`` array:

.. code-block:: text

   moku.data  →  2D array, shape (N, 4) or (N, 9+)

Column Layout (moku.data)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: MAT moku.data columns
   :header-rows: 1
   :widths: 15 10 10 12 35

   * - Column Index
     - Name
     - Type
     - Units
     - Description
   * - 0
     - time
     - float
     - s
     - Timestamp
   * - 1–2
     - —
     - float
     - —
     - Unused (typically 0)
   * - 3
     - phase
     - float
     - cycles
     - Phase in cycles (primary data)
   * - 4–7
     - —
     - float
     - —
     - Unused (if V2 present)
   * - 8
     - V2
     - float
     - cycles
     - Optional second channel phase

**Minimum columns**: 4. If ``moku.data.shape[1] > 8``, column 8 is read as V2.

Post-Processing
~~~~~~~~~~~~~~~

- **Time**: Shifted to start at 0.
- **Phase / V2**: Detrended with ``scipy.signal.detrend(..., type="constant")``.

Validation Rules
~~~~~~~~~~~~~~~~

- Variable ``moku`` must exist.
- ``moku.data`` must be accessible and have at least 4 columns.
- Raises ``ValueError`` if structure is invalid.

Edge Cases and Errors
---------------------

.. list-table:: Error conditions
   :header-rows: 1
   :widths: 35 45

   * - Case
     - Behavior
   * - Non-existent file
     - ``FileNotFoundError``
   * - Unsupported extension
     - ``ValueError`` ("Unsupported file format")
   * - File exceeds size limit
     - ``ValueError`` ("exceeds maximum")
   * - Empty CSV / comments only
     - ``ValueError`` ("No valid data")
   * - CSV with fewer than 4 columns
     - ``ValueError``
   * - MAT missing moku or data
     - ``ValueError`` ("Invalid MAT file structure")

Optional File Size Limit
~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``load()`` does not enforce a file-size cap. Pass ``max_file_size_bytes`` to reject files above a threshold.

Units Summary
-------------

.. list-table:: Units
   :header-rows: 1
   :widths: 15 12 30

   * - Quantity
     - Unit
     - Notes
   * - Time
     - s
     - Seconds
   * - Phase
     - cycles
     - 0–1 = one cycle
   * - Frequency
     - Hz
     - Output of estimators
