import pandas as pd
import matplotlib.pyplot as plt


def plot_summary(summary_csv="results_summary.csv"):
    df = pd.read_csv(summary_csv)

    cost = df.pivot(index="day", columns="controller", values="cost")
    peak = df.pivot(index="day", columns="controller", values="peak")

    print("\nAverage cost by controller:")
    print(cost.mean().sort_values())

    print("\nAverage peak by controller:")
    print(peak.mean().sort_values())

    ax = cost.mean().sort_values().plot(kind="bar")
    ax.set_ylabel("Average cost")
    plt.tight_layout()
    plt.savefig("plot_avg_cost.png")
    plt.close()

    ax = peak.mean().sort_values().plot(kind="bar")
    ax.set_ylabel("Average peak")
    plt.tight_layout()
    plt.savefig("plot_avg_peak.png")
    plt.close()

    ax = cost.plot(marker="o")
    ax.set_ylabel("Cost")
    ax.set_title("Cost per day")
    plt.tight_layout()
    plt.savefig("plot_cost_per_day.png")
    plt.close()

    ax = peak.plot(marker="o")
    ax.set_ylabel("Peak")
    ax.set_title("Peak per day")
    plt.tight_layout()
    plt.savefig("plot_peak_per_day.png")
    plt.close()


def plot_day_timeseries(dqn_csv="results_dqn_day1.csv", rule_csv="results_rule_day1.csv"):
    dqn = pd.read_csv(dqn_csv)
    rule = pd.read_csv(rule_csv)

    plt.plot(dqn["hour"], dqn["load_kw"], label="load")
    plt.plot(rule["hour"], rule["grid_import_kw"], label="rule grid")
    plt.plot(dqn["hour"], dqn["grid_import_kw"], label="dqn grid")
    plt.xlabel("hour")
    plt.ylabel("kW")
    plt.title("Day 1: grid import vs load")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_day1_grid_vs_load.png")
    plt.close()

    plt.plot(rule["hour"], rule["soc_frac"], label="rule soc")
    plt.plot(dqn["hour"], dqn["soc_frac"], label="dqn soc")
    plt.xlabel("hour")
    plt.ylabel("SoC (fraction)")
    plt.title("Day 1: state of charge")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_day1_soc.png")
    plt.close()


if __name__ == "__main__":
    plot_summary("results_summary.csv")
    plot_day_timeseries("results_dqn_day1.csv", "results_rule_day1.csv")
    print("\nSaved plots: plot_avg_cost.png, plot_avg_peak.png, plot_cost_per_day.png, plot_peak_per_day.png, plot_day1_grid_vs_load.png, plot_day1_soc.png")
