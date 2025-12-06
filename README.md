# LCCC-VRP-BP: A Branch-and-Price Solver for Low-Carbon Cold Chain VRP

## 📖 Introduction

This project is the outcome of a group assignment for the "Advanced Operations Research" course at Tongji University.

We address the **Low-Carbon Cold Chain Vehicle Routing Problem with Soft Time Windows (LCCC-VRP-STW)** by constructing a Mixed Integer Programming (MIP) model that considers soft time windows, carbon tax, cargo loss, and refrigeration energy consumption. We developed an exact **Branch-and-Price** algorithm using **Python + Gurobi** to solve this problem.

## ✨ Features

*   **Complex Cost Model**: Comprehensively considers fuel consumption, carbon tax, fresh cargo loss (linear approximation), soft time window penalties, and door opening/closing refrigeration costs.
*   **Exact Algorithm**: Implements a Branch-and-Price framework based on **Column Generation** and **ESPPRC** (Elementary Shortest Path Problem with Resource Constraints).
*   **Efficient Solving**: Capable of finding the theoretical optimal solution (Integer Optimal) for benchmark instances like Solomon C101/R101 within seconds.
*   **Visualization**: Includes a complete route visualization module to intuitively display delivery routes.

## Requirements

*   Python 3.8+
*   **Gurobi Optimizer**: This project uses Gurobi as the LP solver for the Master Problem. You need a valid Gurobi license (Academic or Commercial).
*   Python packages listed in `requirements.txt`:
    *   `numpy`
    *   `matplotlib`
    *   `gurobipy`

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Yinwenxu-1212/LCCC-VRP-BP.git
    cd LCCC-VRP-BP
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Ensure Gurobi is installed and licensed.

## Usage

### 1. Run Single Instance
To run the algorithm on a standard Solomon dataset (e.g., R101):

```bash
python main.py
```
*By default, this runs the algorithm on `dataset/R101.txt` and displays the result.*

### 2. Run Sensitivity Analysis
To analyze the impact of **Carbon Tax**, **Time Windows**, **Freshness Price**, and **Decay Rate**:

```bash
python experiment_sensitivity.py
```
*Results (tables and plots) will be saved in the `output/` directory.*

### 3. Run Scale Experiments
To test the algorithm's performance on different problem scales (e.g., N=25, 50, 100):

```bash
python experiment_scale.py
```
*This uses the `dataset/C110_1.TXT` dataset. Results are saved in `output/`.*

## Project Structure

*   `main.py`: Entry point for single-instance runs.
*   `columnGen.py`: Implements the Column Generation loop (Master Problem).
*   `SPPRC.py`: Solves the Pricing Subproblem (Labeling Algorithm).
*   `branchBound.py`: Implements the Branch-and-Bound tree search.
*   `paramsVRP.py`: Data structures and parameter loading (Cost functions defined here).
*   `route.py`: Route class definition.
*   `solVisualization.py`: Visualization utilities using Matplotlib.
*   `experiment_*.py`: Scripts for batch experiments and sensitivity analysis.
*   `dataset/`: Contains Solomon benchmark instances and other test data.

## References & Acknowledgements

This project builds upon and extends the work from:
*   **[A-Branch-and-Price-Algorithm-for-VRPTW](https://github.com/Guard42/A-Branch-and-Price-Algorithm-for-VRPTW)** by Guard42.

We have significantly modified the core logic to incorporate:
*   **Carbon Tax & Emission logic** in the cost function.
*   **Freshness Decay** cost calculation for cold chain scenarios.
*   **Soft Time Window** handling (versus hard time windows in the original).
*   **Heuristic Pruning** (Beam Search) in the SPPRC solver for performance on larger graphs.
*   **Visualization and Experimentation** scripts for sensitivity analysis.
