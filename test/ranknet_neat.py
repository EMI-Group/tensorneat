###this code will throw a ValueError
from tensorneat import algorithm, genome, common
from tensorneat.pipeline import Pipeline
from tensorneat.genome.gene.node import DefaultNode
from tensorneat.genome.gene.conn import DefaultConn
from tensorneat.genome.operations import mutation
import jax, jax.numpy as jnp
from tensorneat.problem import BaseProblem

def binary_cross_entropy(prediction, target):
    return -(target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction))

# Define the custom Problem
class CustomProblem(BaseProblem):

    jitable = True  # necessary

    def __init__(self, inputs, labels, threshold):
        self.inputs = jnp.array(inputs) #nb! already has shape (n, 768)
        self.labels = jnp.array(labels).reshape((-1,1)) #nb! has shape (n), must be transformed to have shape (n, 1) 
        self.threshold = threshold

        # move the calculation related to pairwise_labels to problem initialization
        pairwise_labels = self.labels - self.labels.T
        self.pairs_to_keep = jnp.abs(pairwise_labels) > self.threshold
        # using nan istead of -inf
        # as any mathmatical operation with nan will result in nan
        pairwise_labels = jnp.where(self.pairs_to_keep, pairwise_labels, jnp.nan)
        self.pairwise_labels = jnp.where(pairwise_labels > 0, True, False)

    
    def evaluate(self, state, randkey, act_func, params):
        # do batch forward for all inputs (using jax.vamp).
        predict = jax.vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )  # should be shape (len(labels), 1)

        #calculating pairwise labels and predictions
        pairwise_predictions = predict - predict.T  # shape (len(inputs), len(inputs))

        pairwise_predictions = jnp.where(self.pairs_to_keep, pairwise_predictions, jnp.nan)
        pairwise_predictions = jax.nn.sigmoid(pairwise_predictions)

        # calculate loss
        loss = binary_cross_entropy(pairwise_predictions, self.pairwise_labels)  # shape (len(labels), len(labels))
        # jax.debug.print("loss={}", loss)
        # reduce loss to a scalar
        # we need to ignore nan value here
        loss = jnp.mean(loss, where=~jnp.isnan(loss))
        # return negative loss as fitness
        # TensorNEAT maximizes fitness, equivalent to minimizing loss
        return -loss

    @property
    def input_shape(self):
        # the input shape that the act_func expects
        return (self.inputs.shape[1],)

    @property
    def output_shape(self):
        # the output shape that the act_func returns
        return (1,)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        # showcase the performance of one individual
        predict = jax.vmap(act_func, in_axes=(None, None, 0))(state, params, self.inputs)

        loss = jnp.mean(jnp.square(predict - self.labels))

        n_elements = 5
        if n_elements > len(self.inputs):
            n_elements = len(self.inputs)

        msg = f"Looking at {n_elements} first elements of input\n"
        for i in range(n_elements):
            msg += f"for input i: {i}, target: {self.labels[i]}, predict: {predict[i]}\n"
        msg += f"total loss: {loss}\n"
        print(msg)

algorithm = algorithm.NEAT(
    pop_size=10,
    survival_threshold=0.2,
    min_species_size=2,
    compatibility_threshold=3.0,  
    species_elitism=2,  
    genome=genome.DefaultGenome(
        num_inputs=768,
        num_outputs=1,
        max_nodes=769,  # must at least be same as inputs and outputs
        max_conns=768,  # must be 768 connections for the network to be fully connected
        output_transform=common.ACT.sigmoid,
        mutation=mutation.DefaultMutation(
            # no allowing adding or deleting nodes
            node_add=0.0,
            node_delete=0.0,
            # set mutation rates for edges to 0.5
            conn_add=0.5,
            conn_delete=0.5,
        ),
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
    ),
)


INPUTS = jax.random.uniform(jax.random.PRNGKey(0), (100, 768)) #the input data x
LABELS = jax.random.uniform(jax.random.PRNGKey(0), (100, )) #the annotated labels y

problem = CustomProblem(INPUTS, LABELS, 0.25)

print("Setting up pipeline and running it")
print("-----------------------------------------------------------------------")
pipeline = Pipeline(
    algorithm,
    problem,
    generation_limit=1,
    fitness_target=1,
    seed=42,
)

state = pipeline.setup()
# run until termination
state, best = pipeline.auto_run(state)
# show results
pipeline.show(state, best)