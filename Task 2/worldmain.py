import world
import pygad
import NeuralNetowork

NN = NeuralNetowork.NeuralNetwork(4)
NN.add_hidden_layer(9)
NN.add_hidden_layer(9)
NN.add_output_layer(4)


def artificial_player(goal_x, goal_y, player_x, player_y):
    answer = NN.get_output([goal_x, goal_y, player_x, player_y])

    for ind, v in enumerate(answer):
        if v > 0.8:
            return ind + 1


def fitness_func(solution):
    NN.set_weights(solution)
    score = world.world(50, artificial_player)
    return score


if __name__ == '__main__':
    num_generations = 1000
    sol_per_pop = 50
    num_parents_mating = 20

    mutation_percent_genes = 10

    parent_selection_type = "sss"

    crossover_type = "single_point"

    mutation_type = "random"

    keep_parents = 5
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
    ga_instance.plot_result()


