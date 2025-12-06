import numpy as np
import time
import os
from branchBound import BranchAndBound
from paramsVRP import ParamsVRP
from route import Route
from solVisualization import solVis

def run_scale_experiment(datasetPath, num_customers, label="", save_fig_path=None):
    # Initialize
    bp = BranchAndBound()
    user_param = ParamsVRP()
    # Limit customers using the new feature in init_params
    user_param.init_params(datasetPath, max_customers=num_customers)
    
    # Initialize Routes
    init_routes = []
    for i in range(user_param.nbclients - 2):
        path = [0, i + 1, user_param.nbclients - 1]
        route_cost = user_param.calculate_actual_cost(path)
        route = Route(path=path, cost=route_cost, Q=1.0)
        init_routes.append(route)
    best_routes = []

    # Run B&P
    start_time = time.time()
    bp.bb_node(user_param, init_routes, None, best_routes, 0)
    end_time = time.time()
    sol_time = end_time - start_time
    
    # Collect Metrics
    total_cost = 0
    total_dist = 0
    total_emission = 0 
    total_fresh_loss = 0
    
    for route in best_routes:
        path = route.get_path()
        total_cost += route.get_cost()
        
        # Recalc components
        current_time = 0.0
        for k in range(len(path) - 1):
            i = path[k]
            j = path[k+1]
            
            d_ij = user_param.dist[i][j]
            # Check for invalid distance (verybig) - same fix as before
            if d_ij >= user_param.verybig / 100:
                d_ij = user_param.dist_base[i][j]
                
            total_dist += d_ij
            
            # Emission: d_ij * rho * eta
            emission = d_ij * user_param.rho_avg * user_param.eta_CO2
            total_emission += emission
            
            # Time update
            if k == 0:
                departure_at_i = 0
            else:
                start_service_at_i = max(current_time, user_param.a[i])
                departure_at_i = start_service_at_i + user_param.s[i]
            
            arrival_at_j = departure_at_i + user_param.ttime[i][j]
            current_time = arrival_at_j
            
            if j != 0 and j != user_param.nbclients - 1:
                # Freshness
                freshness_cost = user_param.P_fresh * user_param.d[j] * user_param.theta * current_time
                total_fresh_loss += freshness_cost

    # Avg Route Length
    avg_dist = total_dist / len(best_routes) if len(best_routes) > 0 else 0
    
    # Visualization
    if save_fig_path:
        dataset_name_with_params = f"C110_1\n(N={num_customers})"
        solVis(user_param, best_routes, sol_time, total_cost, dataset_name_with_params, POPOUT=False, save_path=save_fig_path)
    
    return {
        "scale": num_customers,
        "total_cost": total_cost,
        "time_sec": sol_time,
        "emission": total_emission,
        "num_vehicles": len(best_routes),
        "avg_dist": avg_dist
    }

def run_scale_analysis():
    dataset = "dataset/C110_1.TXT"
    results = []
    
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Scales to test: 25, 50, 100, 200
    # Note: C110_1 might not have 200 customers. Usually Solomon has 100.
    # If 200 is requested but file only has 100, init_params will handle it (min logic).
    scales = [25, 50, 100, 200]
    # Re-run only N=100 for visualization fix
    # scales = [100]
    # Re-run only N=50 for visualization fix
    # scales = [50]
    
    print("=== Experiment 1: Scale Analysis on C110_1 ===")
    
    for n in scales:
        print(f"Running for N = {n} customers...")
        fig_path = f"output/Scale_N{n}.png"
        
        try:
            res = run_scale_experiment(dataset, num_customers=n, label=f"N={n}", save_fig_path=fig_path)
            results.append(res)
            print(f"Done. Cost={res['total_cost']:.2f}, Time={res['time_sec']:.2f}s")
        except Exception as e:
            print(f"Failed for N={n}: {e}")

    # Output Table
    print("\n\n====== Scale Experiment Results Summary ======")
    header = f"{'Scale (N)':<10} | {'Cost':<10} | {'Time(s)':<10} | {'Emission(kg)':<12} | {'Vehicles':<8} | {'Avg Dist':<10}"
    print(header)
    print("-" * 75)
    
    output_file = "output/scale_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("====== Scale Experiment Results Summary ======\n")
        f.write(header + "\n")
        f.write("-" * 75 + "\n")
        
        for r in results:
            line = f"{r['scale']:<10} | {r['total_cost']:<10.2f} | {r['time_sec']:<10.2f} | {r['emission']:<12.2f} | {r['num_vehicles']:<8} | {r['avg_dist']:<10.2f}"
            print(line)
            f.write(line + "\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_scale_analysis()
