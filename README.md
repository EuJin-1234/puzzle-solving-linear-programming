# Solving "The Four Strongest" Mongolian Puzzle via Linear Programming

This project implements a rigorous **Integer Linear Programming (ILP) model** to solve the famous "Four Strongest" Mongolian puzzle. The challenge is to find the arrangement (permutation and orientation) of four distinct animal-shaped toys (Dragon, Tiger, Lion, Garuda) that results in the minimum total length when placed in a row.

Our novel contribution is the development of a **3-Object Inclusion-Exclusion Principle** for the objective function. This approach is demonstrated to be superior to prior pairwise approximation models, yielding a shorter optimal arrangement and providing a more accurate estimation of complex spatial interactions.

## Installation

### Gurobi Setup ###
This project requires the proprietary Gurobi Optimizer.

 * **Gurobi Installation:** Download and install the Gurobi package from the official website.

 * **Gurobi License:** Obtain a non-commercial academic license (free for university students) and configure it on your machine. The Python script relies on a correctly licensed Gurobi installation to run.

Make sure you have Python 3.x installed on your system. You can verify your installation by running:
```bash
python --version
```
Install all the required Python dependencies:
```bash
pip install pandas
pip install gurobipy
```

### Data Setup ###

The model requires the external measurements of the toys.

**Data File:** Ensure the data file, named `distances.xlsx`, which contains the pairwise lengths, is placed in the `data/`.

## Run the Code
```bash
python four_strongest_solver.py
```
