import pygad
import NeuralNetwork

function_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
desired_outputs = [0, 1, 1, 0]

NN = NeuralNetwork.NeuralNetwork(2)
NN.add_hidden_layer(4)
NN.add_hidden_layer(4)
NN.add_output_layer(1)


def fitness_func(solution):
    NN.set_weights(solution)
    error = 0
    for f_input, d_output in zip(function_inputs, desired_outputs):
        output = NN.get_output(f_input)
        error += abs(output[0] - d_output)

    return 1 / (error + 0.0000000001)


if __name__ == "__main__":
    num_generations = 500
    sol_per_pop = 10
    num_parents_mating = 4

    mutation_percent_genes = 10

    parent_selection_type = "sss"

    crossover_type = "single_point"

    mutation_type = "random"

    keep_parents = 1
    num_genes = NN.amount_weights()

    ga_instance = pygad.GA(num_generations=num_generations,
                           sol_per_pop=sol_per_pop,
                           num_parents_mating=num_parents_mating,
                           num_genes=num_genes,
                           fitness_func=fitness_func,
                           mutation_percent_genes=mutation_percent_genes,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           keep_parents=keep_parents,
                           )

    ga_instance.run()
    print(ga_instance.best_solution())

    NN.set_weights(ga_instance.best_solution()[0])
    for f_input in function_inputs:
        print(NN.get_output(f_input))

    ga_instance.plot_result()
