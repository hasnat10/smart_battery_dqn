import pulp
from env import BatteryEnv
from data_loader import load_day_csv


def solve_lp_day(load, price, capacity_kwh=13.5, max_charge_kw=5.0, max_discharge_kw=5.0, init_soc_fraction=0.5, dt_hours=1.0):
    n = len(load)
    model = pulp.LpProblem("battery_lp", pulp.LpMinimize)

    p = [pulp.LpVariable(f"p_{t}", lowBound=-max_discharge_kw, upBound=max_charge_kw) for t in range(n)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=0, upBound=capacity_kwh) for t in range(n + 1)]
    grid = [pulp.LpVariable(f"grid_{t}", lowBound=0) for t in range(n)]

    start_soc = init_soc_fraction * capacity_kwh
    model += soc[0] == start_soc

    for t in range(n):
        model += soc[t + 1] == soc[t] + p[t] * dt_hours
        model += grid[t] >= load[t] + p[t]

    model += sum(grid[t] * price[t] * dt_hours for t in range(n))

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    grid_vals = [grid[t].value() for t in range(n)]
    cost = sum(grid_vals[t] * price[t] * dt_hours for t in range(n))
    peak = max(grid_vals) if grid_vals else 0.0

    return cost, peak


def eval_lp_multi(day_files):
    costs = []
    peaks = []
    for path in day_files:
        load, price = load_day_csv(path)
        c, p = solve_lp_day(load, price)
        costs.append(c)
        peaks.append(p)
    avg_cost = sum(costs) / len(costs)
    avg_peak = sum(peaks) / len(peaks)
    print("lp avg cost:", avg_cost)
    print("lp avg peak:", avg_peak)
