.. _api-reference:

API Reference
=============

Core classes and functions for ring-down signal analysis.

Signal Generation
-----------------

.. autoclass:: ringdownanalysis.signal.RingDownSignal
   :members:
   :undoc-members:
   :show-inheritance:

Frequency Estimators
--------------------

.. autoclass:: ringdownanalysis.estimators.FrequencyEstimator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ringdownanalysis.estimators.NLSFrequencyEstimator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ringdownanalysis.estimators.DFTFrequencyEstimator
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: ringdownanalysis.estimators.EstimationResult

CRLB Calculator
---------------

.. autoclass:: ringdownanalysis.crlb.CRLBCalculator
   :members:
   :undoc-members:
   :show-inheritance:

Data Loading
------------

.. autoclass:: ringdownanalysis.data_loader.RingDownDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

Analysis
--------

.. autoclass:: ringdownanalysis.analyzer.RingDownAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ringdownanalysis.batch_analyzer.BatchRingDownAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ringdownanalysis.batch_analyzer.ProcessResult
   :members:
   :undoc-members:
   :show-inheritance:

Monte Carlo
-----------

.. autoclass:: ringdownanalysis.monte_carlo.MonteCarloAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. autofunction:: ringdownanalysis.configure_logging
