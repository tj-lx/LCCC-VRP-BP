import numpy as np
import time
import os
from branchBound import BranchAndBound
from heuristicGA import GeneticAlgorithm
from paramsVRP import ParamsVRP
from route import Route
from solVisualization import solVis

def calculate_metrics(user_param, routes, solve_time):
    total_cost = 0
    total_dist = 0
    total_emission = 0 
    total_fresh_loss = 0
    
    for route in routes:
        path = route.get_path()
        total_cost += route.get_cost()
        
        # Recalc components
        current_time = 0.0
        for k in range(len(path) - 1):
            i = path[k]
            j = path[k+1]
            
            d_ij = user_param.dist[i][j]
            # Check for invalid distance (verybig)
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
    avg_dist = total_dist / len(routes) if len(routes) > 0 else 0
    
    return {
        "total_cost": total_cost,
        "time_sec": solve_time,
        "emission": total_emission,
        "num_vehicles": len(routes),
        "avg_dist": avg_dist
    }

def run_scale_experiment(datasetPath, num_customers, label="", save_fig_base=None):
    # Initialize Params
    user_param = ParamsVRP()
    user_param.init_params(datasetPath, max_customers=num_customers)
    
    # --- Run B&P ---
    print(f"  > Running B&P for N={num_customers}...")
    bp = BranchAndBound()
    
    # Initialize Routes for B&P
    init_routes = []
    for i in range(user_param.nbclients - 2):
        path = [0, i + 1, user_param.nbclients - 1]
        route_cost = user_param.calculate_actual_cost(path)
        route = Route(path=path, cost=route_cost, Q=1.0)
        init_routes.append(route)
    bp_best_routes = []

    start_time = time.time()
    bp.bb_node(user_param, init_routes, None, bp_best_routes, 0)
    end_time = time.time()
    bp_time = end_time - start_time
    
    bp_metrics = calculate_metrics(user_param, bp_best_routes, bp_time)
    
    if save_fig_base:
        dataset_name_bp = f"C110_1 (B&P)\n(N={num_customers})"
        solVis(user_param, bp_best_routes, bp_time, bp_metrics['total_cost'], 
               dataset_name_bp, POPOUT=False, save_path=f"{save_fig_base}_BP.png")

    # --- Run GA ---
    print(f"  > Running GA for N={num_customers}...")
    # GA parameters can be tuned here if needed. Using defaults.
    # For larger scales, maybe increase generations/pop_size slightly?
    # Keeping defaults for fair comparison baseline.
    ga = GeneticAlgorithm(user_param, pop_size=100, generations=100)
    
    ga_routes, ga_cost_raw, ga_time = ga.run()
    # Note: ga_routes returned by GA.run() are Route objects.
    
    ga_metrics = calculate_metrics(user_param, ga_routes, ga_time)
    
    if save_fig_base:
        dataset_name_ga = f"C110_1 (GA)\n(N={num_customers})"
        solVis(user_param, ga_routes, ga_time, ga_metrics['total_cost'], 
               dataset_name_ga, POPOUT=False, save_path=f"{save_fig_base}_GA.png")
    
    return {
        "scale": num_customers,
        "bp": bp_metrics,
        "ga": ga_metrics
    }

def run_scale_analysis():
    dataset = "dataset/C110_1.TXT"
    results = []
    
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Scales to test: 25, 50, 100, 200
    # Note: 200 scale requires longer runtime
    scales = [25, 50, 100]
    
    print("=== Experiment 1: Scale Analysis (B&P vs GA) on C110_1 ===")
    
    for n in scales:
        print(f"Running for N = {n} customers...")
        fig_base = f"output/Scale_N{n}"
        
        try:
            res = run_scale_experiment(dataset, num_customers=n, label=f"N={n}", save_fig_base=fig_base)
            results.append(res)
            print(f"Done N={n}. B&P Cost={res['bp']['total_cost']:.2f}, GA Cost={res['ga']['total_cost']:.2f}")
        except Exception as e:
            print(f"Failed for N={n}: {e}")
            import traceback
            traceback.print_exc()

    # Output Table
    print("\n\n====== Scale Experiment Results Summary (B&P vs GA) ======")
    # Format: Scale | Method | Cost | Time | Emission | Vehicles
    header = f"{'Scale':<6} | {'Method':<6} | {'Cost':<10} | {'Time(s)':<10} | {'Emission':<10} | {'Vehicles':<8} | {'Avg Dist':<10}"
    print(header)
    print("-" * 80)
    
    output_file = "output/scale_results_comparison.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("====== Scale Experiment Results Summary (B&P vs GA) ======\n")
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            # Print B&P row
            bp = r['bp']
            line_bp = f"{r['scale']:<6} | {'B&P':<6} | {bp['total_cost']:<10.2f} | {bp['time_sec']:<10.2f} | {bp['emission']:<10.2f} | {bp['num_vehicles']:<8} | {bp['avg_dist']:<10.2f}"
            print(line_bp)
            f.write(line_bp + "\n")
            
            # Print GA row
            ga = r['ga']
            line_ga = f"{r['scale']:<6} | {'GA':<6} | {ga['total_cost']:<10.2f} | {ga['time_sec']:<10.2f} | {ga['emission']:<10.2f} | {ga['num_vehicles']:<8} | {ga['avg_dist']:<10.2f}"
            print(line_ga)
            f.write(line_ga + "\n")
            
            # Separator
            print("-" * 80)
            f.write("-" * 80 + "\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_scale_analysis()
