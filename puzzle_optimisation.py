import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os

# --- Model Configuration ---
# Define the set of toys, orientations, and positions
TOYS = ['Dragon', 'Tiger', 'Lion', 'Garuda']
ORIENTATIONS = ['Up', 'Down']
POSITIONS = [1, 2, 3, 4] 

# Individual toy lengths b[toy, orientation] (from Singh's data)
B_LENGTHS = {
    ('Dragon', 'Up'): 10.576,
    ('Tiger', 'Up'): 7.400,
    ('Lion', 'Up'): 6.633,
    ('Garuda', 'Up'): 7.200,
    ('Dragon', 'Down'): 10.576,
    ('Tiger', 'Down'): 7.400,
    ('Lion', 'Down'): 6.633,
    ('Garuda', 'Down'): 7.200,
}

# Threshold for pruning
DISTANCE_THRESHOLD = 0.1 

def load_pairwise_distances(file_name='distances.xlsx'):
    """
    Loads pairwise toy lengths (d) from an external Excel file.
    d[(Toy1, Orient1, Toy2, Orient2)] = combined_length
    """
    try:
        df = pd.read_excel(file_name)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_name}'. Please ensure it is in the same directory.")
        return {}
    
    D_PAIRWISE = {}
    for _, row in df.iterrows():
        key = (row['Toy1'], row['Orientation1'], row['Toy2'], row['Orientation2'])
        D_PAIRWISE[key] = float(row['d'])
    return D_PAIRWISE

def build_and_solve_model(d_pairwise):
    """
    Constructs and solves the Integer Programming model using the 3-Toy 
    Inclusion-Exclusion approach to find the minimum total arrangement length.
    """
    model = gp.Model("FourStrongestPuzzle_3Toy")

    # --- Decision Variables ---
    
    # x[i, j, k]: Binary variable = 1 if toy 'i' is placed at position 'k' with orientation 'j'
    x = model.addVars(
        TOYS, ORIENTATIONS, POSITIONS,
        vtype=GRB.BINARY, name="x"
    )

    # z[i,j,k, i_,j_,k+1, i__,j__,k+2]: Binary variable = 1 if the triplet (i, i_, i__) 
    # is placed consecutively at positions (k, k+1, k+2) with orientations (j, j_, j__).
    z = {}
    count_z_vars = 0
    
    for k in POSITIONS[:-2]: # Iterate through starting positions (1 and 2)
        for i in TOYS:
            for j in ORIENTATIONS:
                for i_ in TOYS:
                    for j_ in ORIENTATIONS:
                        for i__ in TOYS:
                            for j__ in ORIENTATIONS:
                                # 1. Pruning: Check if pairwise lengths are sufficient
                                d1 = d_pairwise.get((i, j, i_, j_), 0)
                                d2 = d_pairwise.get((i_, j_, i__, j__), 0)
                                if d1 < DISTANCE_THRESHOLD or d2 < DISTANCE_THRESHOLD:
                                    continue
                                
                                # 2. Constraint: Ensure all three toys are distinct
                                if i == i_ or i_ == i__ or i == i__:
                                    continue
                                    
                                # Add the z variable
                                z[(i, j, k, i_, j_, i__, j__)] = model.addVar(
                                    vtype=GRB.BINARY,
                                    name=f"z_{i[:1]}{j[:1]}_{k}_{i_[:1]}{j_[:1]}_{k+1}_{i__[:1]}{j__[:1]}_{k+2}"
                                )
                                count_z_vars += 1   

    print(f"INFO: Number of binary triplet variables (z) created: {count_z_vars}")

    # --- Objective Function: Minimize Total Length ---
    # Total Length = Sum(Triplet Lengths) - Sum(Overlaps)
    # Triplet Length T(i, i', i'') = d(i, i') + d(i', i'') - b(i')
    
    internal_positions = POSITIONS[1:-1] # Positions 2 and 3
    
    obj_expr = (
        # Sum of the lengths of all active triplets
        gp.quicksum(
            (d_pairwise.get((i,j,i_,j_),0) + d_pairwise.get((i_,j_,i__,j__),0) - B_LENGTHS[i_,j_]) * z_var
            for (i,j,k,i_,j_,i__,j__), z_var in z.items()
        )
        # Subtract the lengths of the middle toys (B_LENGTHS[i,j]) again, 
        # as they were double-counted (once in each triplet).
        - gp.quicksum(
            B_LENGTHS[i,j] * x[i,j,k]
            for i in TOYS for j in ORIENTATIONS for k in internal_positions
        )
    )

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # --- Constraints ---
    
    # 1. Assignment Constraint (Each position gets exactly one toy)
    for k in POSITIONS:
        model.addConstr(
            gp.quicksum(x[i, j, k] for i in TOYS for j in ORIENTATIONS) == 1,
            name=f"C1_One_Toy_at_Pos_{k}"
        )

    # 2. Uniqueness Constraint (Each toy is used exactly once)
    for i in TOYS:
        model.addConstr(
            gp.quicksum(x[i, j, k] for j in ORIENTATIONS for k in POSITIONS) == 1,
            name=f"C2_Toy_Used_Once_{i}"
        )

    # 3. Triplet Activation (McCormick envelope for z = x1 * x2 * x3)
    for key in z.keys():
        i,j,k,i_,j_,i__,j__ = key
        
        # Upper bounds (z <= x1, z <= x2, z <= x3)
        model.addConstr(z[key] <= x[i,j,k], name=f"C3a_Z_LE_X1_{key}")
        model.addConstr(z[key] <= x[i_,j_,k+1], name=f"C3b_Z_LE_X2_{key}")
        model.addConstr(z[key] <= x[i__,j__,k+2], name=f"C3c_Z_LE_X3_{key}")
        
        # Lower bound (z >= x1 + x2 + x3 - 2)
        model.addConstr(z[key] >= x[i,j,k] + x[i_,j_,k+1] + x[i__,j__,k+2] - 2, 
                        name=f"C3d_Z_GE_X_SUM_{key}")

    # Solve the model
    model.optimize()
    
    # --- Results Handling ---
    if model.Status == GRB.OPTIMAL:
        print("\n--- Optimal Solution Found ---")
        placement = {} 
        for k in POSITIONS:
            for i in TOYS:
                for j in ORIENTATIONS:
                    if x[i, j, k].X > 0.5:
                        placement[k] = (i, j)
                        print(f"Position {k}: {i} ({j})")
        print(f"Minimum Total Length (Objective Value): {model.ObjVal:.3f} cm")
    else:
        print(f"Solver Status: {model.Status}. Optimal solution not found.")


if __name__ == '__main__':
    print("Starting Optimization for 'The Four Strongest' Puzzle...")
    
    # Load data
    D_PAIRWISE = load_pairwise_distances(file_name='distances.xlsx')
    
    if D_PAIRWISE:
        build_and_solve_model(D_PAIRWISE)
