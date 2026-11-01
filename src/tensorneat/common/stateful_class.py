"""Base class for the host-side, state-aware configuration objects.

Dependency DAG position: bottom of ``tensorneat.common``, next to
:mod:`~tensorneat.common.state` (which is its only package import). It is the
common ancestor of ``genome``, ``algorithm`` and ``problem`` classes; nothing
here traces a graph or touches a tensor — a ``StatefulBaseClass`` is
configuration plus a hook that *extends* a ``State`` with its own keys.

Contract for subclasses: ``setup(state)`` is the single place where a
configuration object contributes keys (its own evoxc RNG key splits, its gene
payload shapes, its operator configuration) to the shared ``State``; after
``setup`` the resulting ``State`` is self-contained, and every kernel receives
it explicitly instead of reading attributes off the object during a trace.
"""

from .state import State


class StatefulBaseClass:
    """Mixin for configuration objects that contribute keys to a ``State``.

    Used as ``genome``/``algorithm``/``problem`` base class: the leaf
    configurations (gene types, mutation operators, problems) are
    ``StatefulBaseClass`` instances nested inside a larger one, and
    :meth:`show_config` walks that nesting to produce the run header the
    ``Pipeline`` logs.
    """

    def setup(self, state: State | None = None) -> State:
        """Extend ``state`` with this object's configuration and return it.

        Args:
            state (State | None): The incoming shared state, or ``None`` to
                start from a fresh empty :class:`~tensorneat.common.state.State`.

        Returns:
            State: A new container (``State`` is immutable; ``setup`` never
            mutates its argument) holding every key this object needs at
            build time — configuration values, evoxc RNG keys
            (``evoxc.random.key(seed)`` splits) and any static shapes the
            kernels will be traced against.

        Raises:
            tensorneat.common.errors.ConfigError: If a required configuration
                value is missing or invalid; validation happens here, on the
                host, before any graph is traced.
        """
        if state is None:
            state = State()
        return state

    def show_config(self, registered_objects: list | None = None) -> dict:
        """Collect the (possibly nested) configuration for run logging.

        Args:
            registered_objects (list | None): Accumulator of
                ``StatefulBaseClass`` instances already visited. Pass ``None``
                (the default) at the top-level call; nested
                ``StatefulBaseClass`` attributes are visited by handing the
                same list down, which prevents infinite recursion on shared or
                self-referencing configuration objects.

        Returns:
            dict: A JSON-friendly mapping of attribute name to value, where a
            nested ``StatefulBaseClass`` is rendered as its own nested dict and
            every other value as ``str(value)`` (kept from the pre-rewrite
            library so log headers stay comparable).

        Raises:
            None: Host-only presentation code; it never raises for a value it
            cannot render, and never validates the configuration.
        """
        if registered_objects is None:  # root call
            registered_objects = []

        config = {}
        for key, value in self.__dict__.items():
            if isinstance(value, StatefulBaseClass) and value not in registered_objects:
                registered_objects.append(value)
                config[str(key)] = value.show_config(registered_objects)
            else:
                config[str(key)] = self._render_value(value)
        return config

    @staticmethod
    def _render_value(value) -> str:
        """Render one non-configuration attribute for the run header.

        Args:
            value: Any attribute value of a ``StatefulBaseClass``.

        Returns:
            str: ``str(value)``, or ``"<unrenderable TypeName>"`` when the value's
            own ``__str__`` fails. Logging must not abort a run, and the
            fallback still names the type, so the failure is visible in the
            header rather than swallowed into an empty string.
        """
        try:
            return str(value)
        except Exception:
            return f"<unrenderable {type(value).__name__}>"
