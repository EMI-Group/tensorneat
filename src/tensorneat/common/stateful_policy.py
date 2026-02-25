from typing import Callable, NamedTuple

class PolicyAPI(NamedTuple):
    forward: Callable
    forward_step: Callable
    init_rollout_state: Callable

def _wrap_stateless(as_forward: Callable) -> PolicyAPI:
    def init_rollout_state(state, params):
        return None
    
    def forward_step(state, params, input, rollout_state):
        output = as_forward(state, params, input)
        return output, rollout_state
    
    return PolicyAPI(
        forward=as_forward,
        forward_step=forward_step,
        init_rollout_state=init_rollout_state,
    )