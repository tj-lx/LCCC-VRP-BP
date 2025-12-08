import numpy as np
import time
import os
from branchBound import BranchAndBound
from heuristicGA import GeneticAlgorithm
from paramsVRP import ParamsVRP
from route import Route
from solVisualization import solVis  # Import visualization tool

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
    
    # Cargo Loss Rate (%)
    total_goods_value = sum(user_param.d[j] for j in range(1, user_param.nbclients-1)) * user_param.P_fresh
    loss_rate = (total_fresh_loss / total_goods_value) * 100 if total_goods_value > 0 else 0
    
    return {
        "total_cost": total_cost,
        "time_sec": solve_time,
        "emission": total_emission,
        "num_vehicles": len(routes),
        "avg_dist": avg_dist,
        "loss_rate": loss_rate
    }

def run_single_experiment(datasetPath, c_tax=None, max_lateness=None, p_fresh=None, theta=None, label="", save_fig_base=None):
    # Initialize
    user_param = ParamsVRP()
    user_param.init_params(datasetPath)
    
    # Override parameters if provided
    if c_tax is not None:
        user_param.C_tax = c_tax
    if max_lateness is not None:
        user_param.max_lateness = max_lateness
    if p_fresh is not None:
        user_param.P_fresh = p_fresh
    if theta is not None:
        user_param.theta = theta
        
    # Recalculate static costs because parameters might have changed
    unit_dist_cost = user_param.rho_avg * (user_param.P_fuel + user_param.eta_CO2 * user_param.C_tax) + user_param.beta_ref
    for i in range(user_param.nbclients):
        for j in range(user_param.nbclients):
            user_param.static_cost[i][j] = user_param.dist[i][j] * unit_dist_cost
            user_param.cost[i][j] = user_param.static_cost[i][j]

    # --- Run B&P ---
    print(f"  > Running B&P for {label}...")
    bp = BranchAndBound()
    
    # Initialize Routes
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
        dataset_name_bp = f"R101 (B&P)\n({label})"
        solVis(user_param, bp_best_routes, bp_time, bp_metrics['total_cost'], 
               dataset_name_bp, POPOUT=False, save_path=f"{save_fig_base}_BP.png")

    # --- Run GA ---
    print(f"  > Running GA for {label}...")
    ga = GeneticAlgorithm(user_param, pop_size=100, generations=100)
    ga_routes, ga_cost_raw, ga_time = ga.run()
    
    ga_metrics = calculate_metrics(user_param, ga_routes, ga_time)
    
    if save_fig_base:
        dataset_name_ga = f"R101 (GA)\n({label})"
        solVis(user_param, ga_routes, ga_time, ga_metrics['total_cost'], 
               dataset_name_ga, POPOUT=False, save_path=f"{save_fig_base}_GA.png")

    return {
        "label": label,
        "bp": bp_metrics,
        "ga": ga_metrics
    }

def run_sensitivity_analysis():
    dataset = "dataset/R101.txt"
    results = []
    
    if not os.path.exists("output"):
        os.makedirs("output")

    print("=== Experiment 2.1: Carbon Tax Sensitivity (B&P vs GA) ===")
    tax_levels = [0.0, 0.05, 0.50]
    for tax in tax_levels:
        print(f"Running for C_tax = {tax}...")
        label = f"Tax={tax}"
        fig_base = f"output/VRP_Tax_{tax}"
        res = run_single_experiment(dataset, c_tax=tax, label=label, save_fig_base=fig_base)
        results.append(res)
        print(f"Done. B&P Cost={res['bp']['total_cost']:.2f}, GA Cost={res['ga']['total_cost']:.2f}")

    print("\n=== Experiment 2.2: Time Window Sensitivity (B&P vs GA) ===")
    # Compare Hard TW (max_lateness=0) vs Soft TW (max_lateness=30)
    tw_settings = [(0, "Hard TW"), (30, "Soft TW")]
    for lateness, name in tw_settings:
        print(f"Running for {name} (Lateness={lateness})...")
        label = name
        fig_base = f"output/VRP_TW_{name.replace(' ', '_')}"
        res = run_single_experiment(dataset, c_tax=0.05, max_lateness=lateness, label=label, save_fig_base=fig_base)
        results.append(res)
        print(f"Done. B&P Cost={res['bp']['total_cost']:.2f}, GA Cost={res['ga']['total_cost']:.2f}")

    print("\n=== Experiment 2.3: Freshness Parameters Sensitivity (B&P vs GA) ===")
    
    p_levels = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    theta_levels = [0.001, 0.003, 0.005]
    
    count = 1
    total_exp = len(p_levels) * len(theta_levels)
    
    for p in p_levels:
        for t in theta_levels:
            label = f"P{int(p)}_T{t}"
            print(f"[{count}/{total_exp}] Running for P_fresh={p}, Theta={t}...")
            
            fig_base = f"output/VRP_P{int(p)}_T{t}"
            
            res = run_single_experiment(dataset, p_fresh=p, theta=t, label=label, save_fig_base=fig_base)
            results.append(res)
            print(f"Done. B&P Cost={res['bp']['total_cost']:.2f}, GA Cost={res['ga']['total_cost']:.2f}")
            count += 1

    # Output Table
    print("\n\n====== Experiment Results Summary (B&P vs GA) ======")
    # Format: Scenario | Method | Cost | Emission | Avg Dist | Loss Rate | Vehicles
    header = f"{'Scenario':<15} | {'Method':<6} | {'Cost':<10} | {'Emission':<10} | {'Avg Dist':<10} | {'Loss(%)':<8} | {'Vehicles':<8}"
    print(header)
    print("-" * 85)
    
    output_file = "output/sensitivity_results_comparison.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("====== Experiment Results Summary (B&P vs GA) ======\n")
        f.write(header + "\n")
        f.write("-" * 85 + "\n")
        
        for r in results:
            # B&P
            bp = r['bp']
            line_bp = f"{r['label']:<15} | {'B&P':<6} | {bp['total_cost']:<10.2f} | {bp['emission']:<10.2f} | {bp['avg_dist']:<10.2f} | {bp['loss_rate']:<8.2f} | {bp['num_vehicles']:<8}"
            print(line_bp)
            f.write(line_bp + "\n")
            
            # GA
            ga = r['ga']
            line_ga = f"{r['label']:<15} | {'GA':<6} | {ga['total_cost']:<10.2f} | {ga['emission']:<10.2f} | {ga['avg_dist']:<10.2f} | {ga['loss_rate']:<8.2f} | {ga['num_vehicles']:<8}"
            print(line_ga)
            f.write(line_ga + "\n")
            
            print("-" * 85)
            f.write("-" * 85 + "\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_sensitivity_analysis()
