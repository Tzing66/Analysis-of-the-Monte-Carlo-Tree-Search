# CSE571-Team-Project : Monte Carlo Tree Search
Contributors : Austin Bingham, Jacob Lusenhop, Owin Dieterle, Tanmay Singh

## Requirements and Setup
Python version - 3.11


Libraries - os,random, csv, pandas, math

## Monte Carlo Tree Search

The MCTS algorithm is designed to play the game of Pacman and through this project we aim to demonstrate its efficiency against other search agents - AphaBetaAgent, ExpectimaxAgent, MinimaxAgent which can be used to play the game. Our version of the MCTS incorporates a policy of choosing actions randomly during simulation along with an evaluation function to decide the best move for our pacman.

This project is heavily based on the paper - Browne, Cameron B., et al. "A survey of Monte Carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in games 4.1 (2012) - [link](http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf)

## Commands and Instructions

To run Ms Pacman using Monte Carlo Tree Search without the random policy, use the command `python pacman.py -p MonteCarloAgent -a enableRandomPolicy=0`

To run Ms Pacman using Monte Carlo Tree Search using the random policy and evaluation function, use the command `python pacman.py -p MonteCarloAgent -a enableRandomPolicy=1`

To run the testing script, use the command `python mctstesting.py`

To see the data that we generated and tested, see the "tested data" folder. This folder includes the t tests as well as the normality check. To see other generated data see the "other generated data" folder.
