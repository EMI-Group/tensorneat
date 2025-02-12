# import RankNet
from tensorneat import algorithm, genome, common
from tensorneat.pipeline import Pipeline
from tensorneat.genome import BiasNode
from tensorneat.genome.operations import mutation
from tensorneat.common import ACT, AGG
import jax, jax.numpy as jnp
from tensorneat.problem import BaseProblem

data_num = 100
input_size = 768  # Each network (genome) should have input size 768

# The problem is to optimize a RankNet utilizing NEAT


def binary_cross_entropy(prediction, target):
    return -(target * jnp.log(prediction) + (1 - target) * jnp.log(1 - prediction))


# Create dataset (100 samples of vectors with 768 features)
INPUTS = jax.random.uniform(
    jax.random.PRNGKey(0), (data_num, input_size)
)  # the input data x
LABELS = jax.random.uniform(jax.random.PRNGKey(0), (data_num, 1))  # the annotated labels y
# True (1): >=; False (0): <
pairwise_labels = jnp.where((LABELS - LABELS.T) >= 0, True, False)

print(f"{INPUTS.shape=}, {LABELS.shape=}")


# Define the custom Problem
class CustomProblem(BaseProblem):

    jitable = True  # necessary

    def evaluate(self, state, randkey, act_func, params):
        # Use ``act_func(state, params, inputs)`` to do network forward

        # print("state: ", state)
        # print("params: ",params)
        # print("act_func: ",act_func)

        ans_to_question = True

        # Question: This is the same as doing a forward pass for the generated network?
        #           Meaning the network does 100 passes for all the elements of 768 features?
        if ans_to_question:
            # do batch forward for all inputs (using jax.vamp).
            predict = jax.vmap(act_func, in_axes=(None, None, 0))(
                state, params, INPUTS
            )  # should be shape (100, 1)
        else:
            # I misunderstood, so I have to create a RankNet myself to predict the output
            # Setting up with the values present in the genome
            current_node = state.species.idx2species
            current_node_weights = state.pop_conns[current_node]
            net = RankNet.RankNet(input_size, current_node_weights)
            predict = net.forward(INPUTS)

        pairwise_predictions = predict - predict.T  # shape (100, 100)
        p = jax.nn.sigmoid(pairwise_predictions)  # shape (100, 100)

        # calculate loss
        loss = binary_cross_entropy(p, pairwise_labels)  # shape (100, 100)
        # loss with shape (100, 100), we need to reduce it to a scalar
        loss = jnp.mean(loss)

        # return negative loss as fitness
        # TensorNEAT maximizes fitness, equivalent to minimizing loss
        return -loss

    @property
    def input_shape(self):
        # the input shape that the act_func expects
        return (input_size,)

    @property
    def output_shape(self):
        # the output shape that the act_func returns
        return (1,)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        # showcase the performance of one individual
        predict = jax.vmap(act_func, in_axes=(None, None, 0))(state, params, INPUTS)

        loss = jnp.mean(jnp.square(predict - LABELS))

        msg = ""
        for i in range(INPUTS.shape[0]):
            msg += f"input: {INPUTS[i]}, target: {LABELS[i]}, predict: {predict[i]}\n"
        msg += f"loss: {loss}\n"
        print(msg)


algorithm1 = algorithm.NEAT(
    # setting values to be the same as default in python NEAT package to get same as paper authors
    # tried as best I could to follow this https://neat-python.readthedocs.io/en/latest/config_file.html
    pop_size=100,
    survival_threshold=0.2,
    min_species_size=2,
    species_number_calculate_by="fitness",  # either this or rank, but 'fitness' should be more in line with original paper on NEAT
    # species_size=10, #nothing specified for species_size, it remains default
    # modifying the values the authors explicitly mention
    compatibility_threshold=3.0,  # maybe need to consider this one in the future if weird results, default is 2.0
    species_elitism=2,  # is 2 per default
    genome=genome.DefaultGenome(
        num_inputs=768,
        num_outputs=1,
        max_nodes=769,  # must at least be same as inputs and outputs
        max_conns=768,  # must be 768 connections for the network to be fully connected
        # 0 hidden layers per default
        output_transform=common.ACT.sigmoid,
        mutation=mutation.DefaultMutation(
            # no allowing adding or deleting nodes
            node_add=0.0,
            node_delete=0.0,
            # set mutation rates for edges to 0.5
            conn_add=0.5,
            conn_delete=0.5,
        ),
        node_gene=BiasNode(),
    ),
)

problem = CustomProblem()

pipeline = Pipeline(
    algorithm1,
    problem,
    generation_limit=150,
    fitness_target=1,
    seed=42,
)
state = pipeline.setup()
# run until termination
state, best = pipeline.auto_run(state)
# show results
# pipeline.show(state, best)

network = algorithm1.genome.network_dict(state, *best)