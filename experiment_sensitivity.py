import numpy as np
import time
import os
from branchBound import BranchAndBound
from paramsVRP import ParamsVRP
from route import Route
from solVisualization import solVis  # Import visualization tool

def run_single_experiment(datasetPath, c_tax=None, max_lateness=None, p_fresh=None, theta=None, label="", save_fig_path=None):
    # Initialize
    bp = BranchAndBound()
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
    total_emission = 0 # kg
    total_fresh_loss = 0 # CNY
    
    for route in best_routes:
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
                # Fallback to base distance if current distance is set to infinity (e.g. by branching)
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
                # Freshness: P_fresh * q_j * theta * t_j
                freshness_cost = user_param.P_fresh * user_param.d[j] * user_param.theta * current_time
                total_fresh_loss += freshness_cost

    # Avg Route Length
    avg_dist = total_dist / len(best_routes) if len(best_routes) > 0 else 0
    
    # Cargo Loss Rate (%)
    total_goods_value = sum(user_param.d[j] for j in range(1, user_param.nbclients-1)) * user_param.P_fresh
    loss_rate = (total_fresh_loss / total_goods_value) * 100 if total_goods_value > 0 else 0
    
    # Visualization
    if save_fig_path:
        # Pass the label (which contains params) as dataset_name so it appears in title
        dataset_name_with_params = f"R101\n({label})"
        solVis(user_param, best_routes, sol_time, total_cost, dataset_name_with_params, POPOUT=False, save_path=save_fig_path)
    
    return {
        "label": label,
        "total_cost": total_cost,
        "emission": total_emission,
        "avg_dist": avg_dist,
        "loss_rate": loss_rate,
        "num_vehicles": len(best_routes)
    }

def run_sensitivity_analysis():
    dataset = "dataset/R101.txt"
    results = []
    
    if not os.path.exists("output"):
        os.makedirs("output")

    print("=== Experiment 2.1: Carbon Tax Sensitivity ===")
    tax_levels = [0.0, 0.05, 0.50]
    for tax in tax_levels:
        print(f"Running for C_tax = {tax}...")
        label = f"Tax={tax}"
        fig_path = f"output/VRP_Tax_{tax}.png"
        res = run_single_experiment(dataset, c_tax=tax, label=label, save_fig_path=fig_path)
        results.append(res)
        print(f"Done. Cost={res['total_cost']:.2f}, Saved fig to {fig_path}")

    print("\n=== Experiment 2.2: Time Window Sensitivity ===")
    # Compare Hard TW (max_lateness=0) vs Soft TW (max_lateness=30)
    # Base tax = 0.05
    tw_settings = [(0, "Hard TW"), (30, "Soft TW")]
    for lateness, name in tw_settings:
        print(f"Running for {name} (Lateness={lateness})...")
        label = name
        fig_path = f"output/VRP_TW_{name.replace(' ', '_')}.png"
        res = run_single_experiment(dataset, c_tax=0.05, max_lateness=lateness, label=label, save_fig_path=fig_path)
        results.append(res)
        print(f"Done. Cost={res['total_cost']:.2f}, Saved fig to {fig_path}")

    print("\n=== Experiment 2.3: Freshness Parameters Sensitivity ===")
    
    # P_fresh levels: 10, 20, 30, 40, 50, 60
    p_levels = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    # Theta levels: 0.001, 0.003, 0.005
    theta_levels = [0.001, 0.003, 0.005]
    
    count = 1
    total_exp = len(p_levels) * len(theta_levels)
    
    for p in p_levels:
        for t in theta_levels:
            label = f"P{int(p)}_T{t}"
            print(f"[{count}/{total_exp}] Running for P_fresh={p}, Theta={t}...")
            
            fig_path = f"output/VRP_P{int(p)}_T{t}.png"
            
            res = run_single_experiment(dataset, p_fresh=p, theta=t, label=label, save_fig_path=fig_path)
            results.append(res)
            print(f"Done. Cost={res['total_cost']:.2f}, Saved fig to {fig_path}")
            count += 1

    # Output Table
    print("\n\n====== Experiment Results Summary ======")
    header = f"{'Scenario':<15} | {'Cost':<10} | {'Emission(kg)':<12} | {'Avg Dist(km)':<12} | {'Loss Rate(%)':<12} | {'Vehicles':<8}"
    print(header)
    print("-" * 80)
    
    output_file = "output/sensitivity_results_combined.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("====== Experiment Results Summary ======\n")
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            line = f"{r['label']:<15} | {r['total_cost']:<10.2f} | {r['emission']:<12.2f} | {r['avg_dist']:<12.2f} | {r['loss_rate']:<12.2f} | {r['num_vehicles']:<8}"
            print(line)
            f.write(line + "\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_sensitivity_analysis()
