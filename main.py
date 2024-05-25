import numpy as np
import json

class Config:
    board_size = 3

class TicTacToe:
    def __init__(self, player_one, player_two):
        self.board = np.zeros((Config.board_size, Config.board_size))
        self.player_one = player_one
        self.player_two = player_two
        self.board_hashed = None

        self.game_over = False
        self.player_turn = 1

    def reset_board(self):
        self.board = np.zeros((Config.board_size, Config.board_size))
        self.board_hashed = None
        self.game_over = False
        self.player_turn = 1        

    def hash_board(self):
        self.boardHash = str(self.board.reshape(Config.board_size**2))
        return self.boardHash
    
    def available_positions(self):
        return [(i, j) for i in range(Config.board_size) for j in range(Config.board_size) if self.board[i][j] == 0]

    def update_state(self, position):
        self.board[position] = self.player_turn
        self.player_turn = -1 if self.player_turn == 1 else 1

    def check_winner(self):
        # Check rows
        for i in self.board:
            if all(cell == i[0] and cell != 0 for cell in i):
                return i[0]
        # Check columns
        for j in range(Config.board_size):
            if all(self.board[i][j] == self.board[0][j] and self.board[i][j] != 0 for i in range(Config.board_size)):
                return self.board[0][j]
        # Check diagonals
        if all(self.board[i][i] == self.board[0][0] and self.board[i][i] != 0 for i in range(Config.board_size)):
            return self.board[0][0]
        if all(self.board[i][2 - i] == self.board[0][2] and self.board[i][2 - i] != 0 for i in range(Config.board_size)):
            return self.board[0][2]
        # Check for Draw
        if len(self.available_positions()) == 0:
            return 0
        # Continue the game
        return None
    
    def give_reward(self):
        match self.check_winner():
            case 1:
                self.player_one.reward(1)
                self.player_two.reward(0)
            case -1:
                self.player_one.reward(0)
                self.player_two.reward(1)
            case _:
                self.player_one.reward(0.1)
                self.player_two.reward(0.5)
    
    def render_board(self):
        for i in range(0, Config.board_size):
            print('-------------')
            out = '| '
            for j in range(0, Config.board_size):
                match self.board[i, j]:
                    case 1: token = 'x'
                    case -1: token = 'o'
                    case _: token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

    def train(self, iterations=50):
        for i in range(iterations):
            if i % 5000 == 0:
                print("Rounds {}".format(i))
            while True:
                # Player One
                positions = self.available_positions()
                player_one_actions = self.player_one.action(positions, self.board, self.player_turn)
                self.update_state(player_one_actions)
                self.player_one.add_state(self.hash_board())

                # Check Winner
                if self.check_winner() is not None:
                    self.give_reward()
                    self.player_one.reset()
                    self.player_two.reset()
                    self.reset_board()
                    break
                else:
                    # Player Two
                    positions = self.available_positions()
                    player_two_actions = self.player_two.action(positions, self.board, self.player_turn)
                    self.update_state(player_two_actions)
                    self.player_two.add_state(self.hash_board())

                    # Check Winner
                    if self.check_winner() is not None:
                        self.give_reward()
                        self.player_one.reset()
                        self.player_two.reset()
                        self.reset_board()
                        break

    def play(self):
        while True:
            # Player One
            positions = self.available_positions()
            player_one_actions = self.player_one.action(positions, self.board, self.player_turn)
            self.update_state(player_one_actions)
            self.render_board()

            winner = self.check_winner()
            if winner is not None:
                match winner:
                    case 1: print(self.player_one.name, "wins")
                    case -1: print(self.player_two.name, "wins")
                    case _: print('draw')
                break
            else:
                # Player One
                positions = self.available_positions()
                player_two_actions = self.player_two.action(positions)
                self.update_state(player_two_actions)
                self.render_board()

class Human:
    def __init__(self, name):
        self.name = name
    
    def action(self, positions):
        while True:
            row = int(input("Input your action row:")) - 1
            col = int(input("Input your action col:")) - 1
            action = (row, col)
            if action in positions:
                return action

class Computer:
    def __init__(self, name, learning_rate=0.2, exploaration_prob=0, discount_factor=0.9): #exploaration_prob=0.3
        self.name = name
        self.lr = learning_rate
        self.ep = exploaration_prob
        self.df = discount_factor
        
        self.states = []
        self.states_value = {}

    def action(self, positions, board, symbol):
        if np.random.uniform(0 , 1) < self.ep:
            action = positions[np.random.choice(len(positions))]
        else:
            max_value = -999
            for p in positions:
                cloned_board = board.copy()
                cloned_board[p] = symbol
                cloned_board_hash = self.hash_board(cloned_board)

                if self.states_value.get(cloned_board_hash) is None:
                    value = 0
                else:
                    value = self.states_value.get(cloned_board_hash)
                
                if value > max_value:
                    max_value = value
                    action = p
        return action

    def hash_board(self, board):
        return str(board.reshape(Config.board_size**2))
    
    def add_state(self, state):
        self.states.append(state)

    def reset(self):
        self.states = []
    
    def reward(self, reward):
        for state in reversed(self.states):
            if self.states_value.get(state) is None:
                self.states_value[state] = 0
            self.states_value[state] += self.lr * (self.df * reward - self.states_value[state])
            reward = self.states_value[state]

    def save_model(self, name):
        with open(f'model[{name}].json', 'w') as file:
            json.dump(self.states_value, file)
    
    def load_model(self, name):
        with open(f'model[{name}].json', 'r') as file:
            self.states_value = json.load(file)

if __name__ == '__main__':
    # Training the Computer
    # player_one = Computer("Player 1")
    # player_two = Computer("Player 2")
    # trainer = TicTacToe(player_one, player_two)
    # trainer.train(1000000)
    # player_one.save_model('1000000')

    # Versing the Computer
    player_two = Human("Player 1")
    player_one = Computer("Player 2")
    player_one.load_model('1000000')
    game = TicTacToe(player_one, player_two)
    while True:
        game.reset_board()
        game.play()