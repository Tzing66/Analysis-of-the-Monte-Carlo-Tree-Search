import os
import random
import csv

# Script for randomly generating environments and testing the MCTS and other algorithms.

# Array for different layouts for random selection
layouts = ["capsuleClassic", "contestClassic", "mediumClassic", "openClassic",
           "powerClassic", "smallClassic", "testClassic"]

# Array for different ghost types
ghost_types = ["RandomGhost", "DirectionalGhost"]

# Array for different agents to test
agents = ["MonteCarloAgent -a enableRandomPolicy=0", "MonteCarloAgent -a enableRandomPolicy=1", "MinimaxAgent",
          "AlphaBetaAgent", "ExpectimaxAgent"]

if __name__ == '__main__':
    print("Running tests for Team Project 2")
    num_environments = 3  # how many environments to test on
    games_per_environment = 20  # how many times to run each environment
    tested_environments = []  # tracks environments that have been tested

    # Open the CSV file for writing
    with open('pacman_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers to the CSV file
        writer.writerow(["Layout", "Ghost Type", "Number of Ghosts", "Agent", "Game", "Moves", "Score", "Outcome"])

        for _ in range(num_environments):
            # Select random layouts and number of ghosts
            layout = random.choice(layouts)
            while layout in tested_environments:
                layout = random.choice(layouts)
            num_ghosts = random.randint(1, 5)
            ghost_type = random.choice(ghost_types)
            print(f"Testing environment {layout} with {num_ghosts} {ghost_type} ghosts")

            # Run agent on environment for number of games
            for game in range(1, games_per_environment + 1):
                for agent in agents:
                    try:
                        print(f"Testing agent {agent} in game {game}")
                        command = f'python pacman.py -l {layout} -p {agent} -g {ghost_type} -k {num_ghosts} -n 1 --quietTextGraphics'
                        result = os.popen(command).read()
                        print("Result captured: ")
                        print(result)

                        # Extract relevant information from the result
                        lines = result.split("\n")
                        moves = lines[-3].split(":")[1].strip()
                        score = lines[0].split(":")[1].strip()

                        # Check if Pacman emerges victorious
                        if "Pacman emerges victorious" in result:
                            outcome = '1'
                        else:
                            outcome = '0'

                        # Write the results to the CSV file
                        writer.writerow([layout, ghost_type, num_ghosts, agent, game, moves, score, outcome])
                    except FileNotFoundError:
                        print("pacman file not found")