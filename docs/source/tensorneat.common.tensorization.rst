tensorneat.common.tensorization module
--------------------------------------

.. automodule:: tensorneat.common.tensorization
   :members:
   :show-inheritance:

.. ``derive_schedule`` is an ``evoxc.trace.defn.Defn`` instance, so the
   ``automodule`` above never registers it: autodoc drops a member whose
   ``__module__`` ("evoxc.trace.defn") is not this module's.  This directive is
   its only registration; the ``(*args, **kwargs)`` signature it renders is the
   wrapper's, because ``Defn`` exposes no ``__signature__``, so the real
   parameters are the ``Args:`` block below.
.. autofunction:: tensorneat.common.tensorization.derive_schedule
