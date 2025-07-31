import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import Image, display

# -----------------------
# Helper Functions
# -----------------------

def fitness(board):
  """
  Calculates the fitness of a given board.
  Fitness = negative number of attacking pairs of queens.
  The closer the value is to zero, the fewer conflicts there are.
  i.e., Higher fitness is better.
  """
  n = len(board)
  attacks = 0
  for i in range(n):
    for j in range(i+1, n):
      # Queens attack if on same row or diagnal
      if board[i] == board[j] or abs(board[i] - board[j]) == abs(i-j):
        attacks += 1
  return -attacks   # minimize conflicts

def random_solution(n):
  """
  Generates a random solution for the N-Queens problem.
  Each column is assigned a random row for the queen.
  """
  return [random.randint(0, n-1) for _ in range(n)]

def generate_neighbors(board, num_neighbors=3):
  """
  Creates neighboring solutions by randomly moving a queen in one column.
  num_neighbors controls how many neighbors to generate.
  """
  neighbors = []
  n = len(board)
  for _ in range(num_neighbors):
    new_board = board[:]
    col = random.randint(0, n-1)
    new_board[col] = random.randint(0, n-1)
    neighbors.append(new_board)
  return neighbors

def center_of_mass(population, fitnesses):
  """
  Computes the 'center of mass' solution.
  Uses weighted average of population positions based on fitness.
  This guides the algorithm towards promising regions of the search space.
  """
  fitness_shifted = [f - min(fitnesses) + 1e-6 for f in fitnesses]
  weights = np.array(fitness_shifted)
  boards = np.array(population)
  return np.rint(np.average(boards, axis=0, weights=weights)).astype(int).tolist()

def local_search(board):
  """
  Improves a solution locally by testing neighboring boards and keeping the best one.
  This adds a hill-climbing aspect to BB/BC (improving step by step instead of making large jumps).
  """
  best_board = board[:]
  best_fit = fitness(board)
  for neighbor in generate_neighbors(board, num_neighbors=10):
    f = fitness(neighbor)
    if f > best_fit:
      best_board, best_fit = neighbor, f
  return best_board

# -----------------------------
# BB/BC Algorithm for N-Queens
# -----------------------------

def bbbc_nqueens(n=8, population_size=30, max_cycles=100):
  """
  Solves N-Queens using Big Bang / Big Crunch optimization.

  Big Bang phase: generates population randomly or from elite solutions
  Big Crunch phase: repeatedly contracts population size by focusing around a center of mass.
  """
  elite_pool = [] # Stores good solutions found so far
  best_solution = None
  best_fitness = float('-inf')  # Initializes worst fitness
  history = []  # Stores (solution, fitness) for convergence visualization and animation

  """
  max_cycles controls how many Big Bang / Big Crunch cycles occur.
  Each cycle:
    Starts with a new population (from scratch or elite pool)
    Runs until the population contracts to 1 solution
  """
  for _ in range(max_cycles):
    # Big Bang Phase
    # Step 1: Generate population
    if not elite_pool:
      population = [random_solution(n) for _ in range(population_size)]
    else:
      population = [random.choice(elite_pool)[:] for _ in range(population_size)]
      for p in population:
        if random.random() < 0.5:
          p[random.randint(0, n-1)] = random.randint(0, n-1)

    # Big Crunch Phase
    while len(population) > 1:
      fitnesses = [fitness(ind) for ind in population]

      # Step 2: Neighborhood search
      new_population = []
      for ind in population:
        neighbors = generate_neighbors(ind)
        best_neighbor = max(neighbors+[ind], key=fitness)
        new_population.append(best_neighbor)
      population = new_population

      # Step 3: Find center of mass
      center = center_of_mass(population, fitnesses)

      # Step 4: Local search on center
      center = local_search(center)

      # Step 5: Update elite pool
      elite_pool.append(center)

      # Track best solution
      center_fit = fitness(center)
      history.append((center[:], center_fit))
      if center_fit > best_fitness:
        best_solution, best_fitness = center, center_fit

      # Step 6: Eliminate poor solutions
      sorted_population = sorted(population, key=fitness, reverse=True)
      population = sorted_population[: max(1, len(population) // 2)]

      if best_fitness == 0:
        return best_solution, history

  return best_solution, history

# -------------------------
# Visualization Functions
# -------------------------

def plot_solution(board):
  """
  Plots a chessboard and places queens on it using Unicode characters
  """
  n = len(board)
  board_matrix = np.zeros((n, n))
  for col, row in enumerate(board):
    board_matrix[row, col] = 1

  fig, ax = plt.subplots()
  ax.imshow([[ (i+j) % 2 for j in range(n)] for i in range(n)], cmap='binary')
  for col, row in enumerate(board):
    ax.text(col, row, '♛', ha='center', va='center', fontsize=20, color='red')

  ax.set_xticks(range(n))
  ax.set_yticks(range(n))
  plt.gca().invert_yaxis() # Flip axis so (0,0) is bottom-left
  plt.show()

def animate_solution(history, filename='bbbc_solution.gif'):
  """
  Creates a GIF animation showing how the solution evolves over time.
  """
  fig, ax = plt.subplots()

  def update(frame):
    board, _ = history[frame]
    ax.clear()
    ax.imshow([[ (i+j) % 2 for j in range(len(board))] for i in range(len(board))], cmap='binary')
    for col, row in enumerate(board):
      ax.text(col, row, '♛', ha='center', va='center', fontsize=20, color='red')
    ax.set_xticks(range(len(board)))
    ax.set_yticks(range(len(board)))
    ax.invert_yaxis()
    ax.set_title(f'Iteration {frame}')

  ani = animation.FuncAnimation(fig, update, frames=len(history), interval=300, repeat=False)
  ani.save(filename, writer='imagemagick')
  plt.close(fig)
  display(Image(filename=filename))
  return filename

def plot_convergence(history):
  """
  Plots fitness values for each iteration to visualize convergence.
  Higher fitness (closer to 0) indicates fewer queen conflicts.
  Max Iterations = number of cycle restarts.
  Convergence plot length = total number of contraction steps across all restarts.
  """
  fitness_values = [fit for _, fit in history]
  plt.figure()
  plt.plot(range(len(fitness_values)), fitness_values, marker='o')
  plt.title('Convergence Over Iterations')
  plt.xlabel('Iteration')
  plt.ylabel('Fitness (negative conflicts)')
  plt.grid(True)
  plt.show()
