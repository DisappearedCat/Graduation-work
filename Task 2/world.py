import random

random.seed()

world_size = {'x': 7, 'y': 7}


def world(live_time, get_turn):
    """
    Main function of the world

    :param live_time: number of round that world exist
    :param get_turn: function that return player's move. Must accept 4 numbers:  goal's and self coordinates.
    Must return number from 1 to 4 or -1. 1 - up, 2 - down, 3 - right, 4 - left, -1 - stay
    Example: turn = get_turn(goals_x, goals_y, self_x, self_y)
    :return: Scores that player gain
    """

    player = {'x': world_size['x'] // 2, 'y': world_size['y'] // 2, 'score': 0}
    goal = {'x': random.randint(0, world_size['x'] - 1), 'y': random.randint(0, world_size['y'] - 1)}

    for i in range(live_time):
        turn = get_turn(goal['x'], goal['y'], player['x'], player['y'])

        if turn == 2 and player['y'] < (world_size['y'] - 1):  # down
            player['y'] += 1
        elif turn == 1 and player['y'] > 0:  # up
            player['y'] -= 1
        elif turn == 3 and player['x'] < (world_size['x'] - 1):  # right
            player['x'] += 1
        elif turn == 4 and player['x'] > 0:  # left
            player['x'] -= 1

        if goal['x'] == player['x'] and goal['y'] == player['y']:
            goal['x'] = random.randint(0, world_size['x'] - 1)
            goal['y'] = random.randint(0, world_size['y'] - 1)
            player['score'] += 1

    return player['score']


if __name__ == "__main__":
    def human_player(goal_x, goal_y, player_x, player_y):
        for i in range(world_size['y']):
            for j in range(world_size['x']):
                if i == goal_y and j == goal_x:
                    print('X', end=' ')
                elif i == player_y and j == player_x:
                    print('P', end=' ')
                else:
                    print('O', end=' ')
            print()

        turn = input("Print turn:")
        return int(turn)


    print(world(30, human_player))
