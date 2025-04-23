# recurrent_mutation.py
import jax
import jax.numpy as jnp

# same helpers as DefaultMutation
from tensorneat.common import fetch_first, I_INF, check_cycles, fetch_random
from tensorneat.genome.operations.mutation.default import DefaultMutation
from tensorneat.genome.utils import (
    extract_gene_attrs,
    add_conn,
    add_node,
    unflatten_conns,
    delete_conn_by_pos,
    delete_node_by_pos,
    set_gene_attrs, 
    )

class RecurrentMutation(DefaultMutation):
    """
    A DefaultMutation variant that *biases* (not forces) evolution toward
    recurrent connections, controlled by the parameter "p_recur".

    Applies add/delete node and delete connection mutations identically to
    the user's original implementation provided, but uses an optimized
    add-connection strategy with cycle bias and retries.

    Parameters
    ----------
    p_recur : float, default 0.1
        Probability (0 to 1) that any single *successful* add-connection
        mutation *must* result in forming a cycle in the graph.
    max_conn_tries : int, default 20
        Maximum number of distinct (source, target) node pairs to sample
        when attempting the add-connection mutation before giving up for
        this genome in this generation.
    conn_add : float
        Probability of attempting connection addition. Inherited from 
        DefaultMutation.
    conn_delete : float
        Probability of attempting connection deletion. Inherited from 
        DefaultMutation.
    node_add : float
        Probability of attempting node addition. Inherited from DefaultMutation.
    node_delete : float
        Probability of attempting node deletion. Inherited from DefaultMutation.
    kwargs : Other arguments accepted by DefaultMutation.__init__
    """
    # constructor                                                        
    def __init__(
        self,
        *,
        p_recur: float = 0.1,
        max_conn_tries: int = 20,
        **kwargs        # Pass conn_add, conn_delete etc. through to base class
        ):
        # Initialize base class probabilities from DefaultMutation
        super().__init__(**kwargs)

        if not 0.0 <= p_recur <= 1.0:
            raise ValueError("p_recur must be in [0, 1]")
        if max_conn_tries < 1:
            raise ValueError("max_conn_tries must be >= 1")

        self.p_recur        = float(p_recur)
        self.max_conn_tries = int(max_conn_tries)

    # structural mutation (all value‑mutation logic is inherited intact) 
    def mutate_structure(
        self, state, genome, randkey, nodes, conns, new_node_key, new_conn_key
        ):
        """
        Apply ONE structural mutation (node-add/del, conn-add/del) chosen
        according to the probabilities stored in this object.
        Uses the custom `_mutate_add_conn_recurrent_optimized`.
        Uses user's original implementations for add_node, delete_node, 
        delete_conn.
        """
        # Helper functions (kept as per user request, structure preserved
        def mutate_add_node(key_, nodes_, conns_):
            remain_node_space = jnp.isnan(nodes_[:, 0]).sum()
            remain_conn_space = jnp.isnan(conns_[:, 0]).sum()
          
            i_key, o_key, idx = self.choose_connection_key(key_, conns_)

            def successful_add_node():
                # remove the original connection and record its attrs
                original_attrs = extract_gene_attrs(
                    genome.conn_gene, conns_[idx]
                    )
                new_conns = delete_conn_by_pos(conns_, idx)

                # add a new node with identity attrs
                new_nodes = add_node(
                    nodes_, 
                    jnp.array([new_node_key]), 
                    genome.node_gene.new_identity_attrs(state)
                    )

                # build two replacement connections
                if "historical_marker" in genome.conn_gene.fixed_attrs:
                    f1 = jnp.array([i_key, new_node_key, new_conn_key[0]])
                    f2 = jnp.array([new_node_key, o_key, new_conn_key[1]])
                else:
                    f1 = jnp.array([i_key, new_node_key])
                    f2 = jnp.array([new_node_key, o_key])
                
                new_conns = add_conn(
                    new_conns, 
                    f1, 
                    genome.conn_gene.new_identity_attrs(state)
                    )
                new_conns = add_conn(new_conns, f2, original_attrs)
                return new_nodes, new_conns

            cond_do_nothing = (idx == I_INF) | \
                              (remain_node_space < 1) | \
                              (remain_conn_space < 2)
            return jax.lax.cond(
                cond_do_nothing,            # condition for doing nothing
                lambda: (nodes_, conns_),   # do nothing branch (if cond is true)
                successful_add_node         # do add branch (if cond is false)
                )

        def mutate_delete_node(key_, nodes_, conns_):
            k, idx = self.choose_node_key(
                key_, nodes_, genome.input_idx, genome.output_idx,
                allow_input_keys=False, allow_output_keys=False,
                )
            
            def do():
                # delete the node
                new_nodes = delete_node_by_pos(nodes_, idx)
                
                # delete all connections
                new_conns = jnp.where(
                    ((conns_[:, 0] == k) | (conns_[:, 1] == k))[:, None],
                    jnp.nan, conns_
                    )
                return new_nodes, new_conns
                
            return jax.lax.cond(
                idx == I_INF,               # cond to determine "doing nothing" 
                lambda: (nodes_, conns_),   # Do nothing branch
                do                          # do delete branch
                )

        # NEW biased add‑connection with optimizations and p_recur bias
        def _mutate_add_conn_recurrent_optimized(key_, nodes_, conns_):
            """
            Optimized: Attempts to insert one connection with recurrent bias.
            Makes up to "self.max_conn_tries" draws.
            Computes graph structure once before vmap.
            Checks prerequisites before cycle check.
            """
            remain_conn_space = jnp.isnan(conns_[:, 0]).sum()
            space_available = remain_conn_space >= 1

            # Optimization: Calculate graph structure ONCE before vmap 
            u_conns = unflatten_conns(nodes_, conns_)
            conns_exist_matrix = u_conns != I_INF

            # Function for a single attempt 
            def attempt(k_triplet):
                """One sampling attempt; returns (accept?, i_key, o_key)."""
                k1_node, k2_node, k_recur = jax.random.split(k_triplet, 3)

                # 1. Sample endpoints
                i_key, from_idx = self.choose_node_key(
                    k1_node, nodes_, genome.input_idx, genome.output_idx,
                    allow_input_keys=True, allow_output_keys=True,
                    )
                o_key, to_idx = self.choose_node_key(
                    k2_node, nodes_, genome.input_idx, genome.output_idx,
                    allow_input_keys=False, allow_output_keys=True,
                    )

                # 2. Basic checks: nodes selected? space available?
                nodes_chosen = (from_idx != I_INF) & (to_idx != I_INF)
                prereqs_met = nodes_chosen & space_available

                # --- Nested conditional logic for efficiency ---
                def check_existence_and_cycle():
                    # 3. Duplicate check (only if prereqs met)
                    exists = fetch_first(
                        (conns_[:, 0] == i_key) & (conns_[:, 1] == o_key)
                        ) != I_INF
                    not_duplicate = ~exists

                    def check_cycle_logic():
                        # 4. Cycle check (only if valid, non-duplicate candidate)
                        forms_cy = check_cycles(
                            nodes_, 
                            conns_exist_matrix, 
                            from_idx, to_idx
                            )

                        # 5. Decide if we *require* a cycle
                        force_recur = jax.random.uniform(k_recur) < self.p_recur

                        # 6. Final check: cycle requirement satisfaction
                        cycle_req_satisfied = (force_recur & forms_cy) | \
                                              (~force_recur)

                        # Attempt is valid if cycle requirement is met
                        is_valid = cycle_req_satisfied
                        return is_valid, i_key, o_key

                    # If not duplicate, check cycle logic, else invalid
                    is_valid, final_i_key, final_o_key = jax.lax.cond(
                        not_duplicate,
                        check_cycle_logic,
                        lambda: (jnp.array(False), jnp.nan, jnp.nan) # is_valid=False if duplicate
                    )
                    return is_valid, final_i_key, final_o_key

                # If basic prereqs met, check existence/cycle, else invalid
                is_valid_attempt, i_key_attempt, o_key_attempt = jax.lax.cond(
                    prereqs_met,
                    check_existence_and_cycle,
                    lambda: (jnp.array(False), jnp.nan, jnp.nan) # is_valid=False if prereqs not met
                    )

                return is_valid_attempt, i_key_attempt, o_key_attempt # bool, float, float

            # Vectorize attempts and find first success
            keys = jax.random.split(key_, self.max_conn_tries)
            accept_flags, i_keys_all, o_keys_all = jax.vmap(attempt)(keys)

            # find FIRST successful draw (if any)
            first_success_idx = fetch_first(accept_flags)
            found_valid_candidate = first_success_idx != I_INF

            # Conditionally add the connection
            def do_accept():
                # Get the keys from the first successful attempt
                i_key_chosen = i_keys_all[first_success_idx]
                o_key_chosen = o_keys_all[first_success_idx]
                
                # Create fixed attributes (using 3rd marker for add_conn)
                if "historical_marker" in genome.conn_gene.fixed_attrs:
                    fix = jnp.array([i_key_chosen, o_key_chosen, new_conn_key[2]])
                else:
                    fix = jnp.array([i_key_chosen, o_key_chosen])

                # Add the connection
                new_conns = add_conn(
                    conns_,
                    fix,
                    genome.conn_gene.new_zero_attrs(state) # Add with zero/default attrs
                    )

                return nodes_, new_conns
            
            # If a valid candidate was found, add it; otherwise return original 
            # arrays
            return jax.lax.cond(
                found_valid_candidate,   # Condition is True if we should accept
                do_accept,               # Do accept branch (if cond is true)
                lambda: (nodes_, conns_) # Do nothing branch (if cond is false)
                )

        def mutate_delete_conn(key_, nodes_, conns_):        
            i_key, o_key, idx = self.choose_connection_key(key_, conns_)
            return jax.lax.cond(
                idx == I_INF,
                lambda: (nodes_, conns_),
                lambda: (nodes_, delete_conn_by_pos(conns_, idx)),
                )

        # --- Scheduling Logic (unchanged from user's code) ---
        k_node_add, k_node_del, k_conn_add, k_conn_del, k_schedule = \
            jax.random.split(randkey, 5)
        probs = jax.random.uniform(k_schedule, (4,))

        def nothing(_, n_, c_): return n_, c_

        # Apply mutations conditionally using original helper functions
        nodes, conns = jax.lax.cond(
            (self.node_add > 0) & (probs[0] < self.node_add),
            mutate_add_node, nothing, k_node_add, nodes, conns
            )
        nodes, conns = jax.lax.cond(
            (self.node_delete > 0) & (probs[1] < self.node_delete),
            mutate_delete_node, nothing, k_node_del, nodes, conns
            )
        # Apply connection addition using the OPTIMIZED version
        nodes, conns = jax.lax.cond(
            (self.conn_add > 0) & (probs[2] < self.conn_add),
            _mutate_add_conn_recurrent_optimized, # <<< Use optimized add_conn
             nothing, k_conn_add, nodes, conns
            )
        # Apply connection deletion using original helper function
        nodes, conns = jax.lax.cond(
            (self.conn_delete > 0) & (probs[3] < self.conn_delete),
            mutate_delete_conn, nothing, k_conn_del, nodes, conns
             )
        return nodes, conns

