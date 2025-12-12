from lp_baseline import eval_lp_multi, solve_lp_day
import random
import csv
from env import BatteryEnv
from dqn_agent import DQNAgent
from data_loader import load_day_csv



DAY_FILES = [
    "data/day1.csv",
    "data/day2.csv",
    "data/day3.csv",
    "data/day4.csv",
    "data/day5.csv",
    "data/day6.csv",
]


def eval_no_battery_multi():
    costs = []
    peaks = []
    for path in DAY_FILES:
        load, price = load_day_csv(path)
        day_cost = 0.0
        day_peak = 0.0
        for p_load, p_price in zip(load, price):
            day_cost += p_load * p_price
            if p_load > day_peak:
                day_peak = p_load
        costs.append(day_cost)
        peaks.append(day_peak)
    avg_cost = sum(costs) / len(costs)
    avg_peak = sum(peaks) / len(peaks)
    print("no-battery avg cost:", avg_cost)
    print("no-battery avg peak:", avg_peak)

def record_dqn_day(agent, in_path, out_path):
    load, price = load_day_csv(in_path)
    env = BatteryEnv(load, price, use_forecast=True)
    state = env.reset()
    done = False
    rows = []
    t = 0
    while not done:
        a = agent.select_greedy(state)
        next_state, r, done, info = env.step(a)
        rows.append(
            [
                t,
                state[1],
                state[2],
                state[3],
                a,
                info["grid_import_kw"],
                info["step_cost"],
            ]
        )
        state = next_state
        t += 1
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "hour",
                "soc_frac",
                "price",
                "load_kw",
                "action",
                "grid_import_kw",
                "step_cost",
            ]
        )
        w.writerows(rows)
    print("saved dqn day to", out_path)

def record_rule_day(in_path, out_path):
    load, price = load_day_csv(in_path)
    env = BatteryEnv(load, price)
    state = env.reset()
    done = False
    rows = []
    t = 0
    while not done:
        a = rule_policy(state)
        next_state, r, done, info = env.step(a)
        rows.append(
            [
                t,
                state[1],
                state[2],
                state[3],
                a,
                info["grid_import_kw"],
                info["step_cost"],
            ]
        )
        state = next_state
        t += 1
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "hour",
                "soc_frac",
                "price",
                "load_kw",
                "action",
                "grid_import_kw",
                "step_cost",
            ]
        )
        w.writerows(rows)
    print("saved rule day to", out_path)




def rule_policy(state):
    hour, soc, price, load = state
    low_price = 0.15
    high_price = 0.3
    low_soc = 0.2
    high_soc = 0.8
    if price <= low_price and soc < high_soc:
        return 0
    if price >= high_price and soc > low_soc:
        return 2
    return 1


def eval_rule_multi():
    costs = []
    peaks = []
    for path in DAY_FILES:
        load, price = load_day_csv(path)
        env = BatteryEnv(load, price)
        state = env.reset()
        done = False
        while not done:
            a = rule_policy(state)
            state, r, done, info = env.step(a)
        costs.append(info["total_cost"])
        peaks.append(info["peak_import_kw"])
    avg_cost = sum(costs) / len(costs)
    avg_peak = sum(peaks) / len(peaks)
    print("rule avg cost:", avg_cost)
    print("rule avg peak:", avg_peak)


def train_dqn(episodes=300, use_forecast=True, seed=None):
    if seed is not None:
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except:
            pass
        try:
            import torch
            torch.manual_seed(seed)
        except:
            pass

    action_dim = 3
    sample_load, sample_price = load_day_csv(DAY_FILES[0])
    sample_env = BatteryEnv(sample_load, sample_price, use_forecast=use_forecast)
    state_dim = len(sample_env.reset())

    agent = DQNAgent(state_dim, action_dim)

    for ep in range(episodes):
        path = random.choice(DAY_FILES)
        load, price = load_day_csv(path)
        env = BatteryEnv(load, price, use_forecast=use_forecast)
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            a = agent.select_action(state)
            next_state, r, done, info = env.step(a)
            agent.store(state, a, r, next_state, float(done))
            agent.train_step()
            state = next_state
            ep_reward += r

            if (ep + 1) % 50 == 0:
                print("episode", ep + 1, "reward", ep_reward, "cost", info["total_cost"])

    return agent

def run_dqn_seeds(seeds=(0, 1, 2, 3, 4), episodes=300):
    rows = []
    for seed in seeds:
        print("\n=== seed", seed, "===")
        agent = train_dqn(episodes=episodes, use_forecast=True, seed=seed)

        costs = []
        peaks = []
        for path in DAY_FILES:
            load, price = load_day_csv(path)
            env = BatteryEnv(load, price, use_forecast=True)
            state = env.reset()
            done = False
            while not done:
                a = agent.select_greedy(state)
                state, r, done, info = env.step(a)
            costs.append(info["total_cost"])
            peaks.append(info["peak_import_kw"])

        avg_cost = sum(costs) / len(costs)
        avg_peak = sum(peaks) / len(peaks)
        print("seed", seed, "avg_cost", avg_cost, "avg_peak", avg_peak)
        rows.append([seed, avg_cost, avg_peak])

    with open("results_dqn_seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "avg_cost", "avg_peak"])
        w.writerows(rows)

    print("saved to results_dqn_seeds.csv")



def eval_dqn_multi(agent, use_forecast=True):
    costs = []
    peaks = []
    for path in DAY_FILES:
        load, price = load_day_csv(path)
        env = BatteryEnv(load, price, use_forecast=use_forecast)
        state = env.reset()
        done = False
        while not done:
            a = agent.select_greedy(state)
            state, r, done, info = env.step(a)
        costs.append(info["total_cost"])
        peaks.append(info["peak_import_kw"])

    avg_cost = sum(costs) / len(costs)
    avg_peak = sum(peaks) / len(peaks)
    print("dqn avg cost:", avg_cost)
    print("dqn avg peak:", avg_peak)


def write_summary(agent, out_path="results_summary.csv"):
    rows = []
    for idx, path in enumerate(DAY_FILES, start=1):
        day_name = "day" + str(idx)
        load, price = load_day_csv(path)

        day_cost = 0.0
        day_peak = 0.0
        for p_load, p_price in zip(load, price):
            day_cost += p_load * p_price
            if p_load > day_peak:
                day_peak = p_load
        rows.append([day_name, "no_battery", day_cost, day_peak])

        env = BatteryEnv(load, price)
        state = env.reset()
        done = False
        while not done:
            a = rule_policy(state)
            state, r, done, info = env.step(a)
        rows.append([day_name, "rule", info["total_cost"], info["peak_import_kw"]])

        c_lp, p_lp = solve_lp_day(load, price)
        rows.append([day_name, "lp", c_lp, p_lp])

        env = BatteryEnv(load, price, use_forecast=True)
        state = env.reset()
        done = False
        while not done:
            a = agent.select_greedy(state)
            state, r, done, info = env.step(a)
        rows.append([day_name, "dqn", info["total_cost"], info["peak_import_kw"]])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "controller", "cost", "peak"])
        w.writerows(rows)
    print("saved summary to", out_path)



if __name__ == "__main__":
    eval_no_battery_multi()
    eval_rule_multi()
    eval_lp_multi(DAY_FILES)
    run_dqn_seeds(seeds=(0, 1, 2, 3, 4), episodes=300)
    agent = train_dqn(episodes=300, use_forecast=True, seed=0)
    eval_dqn_multi(agent, use_forecast=True)
    record_dqn_day(agent, "data/day1.csv", "results_dqn_day1.csv")
    record_rule_day("data/day1.csv", "results_rule_day1.csv")
    write_summary(agent, "results_summary.csv")






