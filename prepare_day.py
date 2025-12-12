import pandas as pd


def build_day_csv(src_path, dst_path, day_index):
    df = pd.read_csv(src_path)
    start = day_index * 24
    end = start + 24
    day_df = df.iloc[start:end].copy()

    prices = [0.10] * 8 + [0.20] * 8 + [0.40] * 8
    prices = prices[: len(day_df)]

    out = pd.DataFrame(
        {
            "load_kw": day_df["consumption"].values,
            "price": prices,
        }
    )
    out.to_csv(dst_path, index=False)


if __name__ == "__main__":
    src = "raw/energy_consumption_levels.csv"
    build_day_csv(src, "data/day1.csv", day_index=0)
    build_day_csv(src, "data/day2.csv", day_index=1)
    build_day_csv(src, "data/day3.csv", day_index=2)
    build_day_csv(src, "data/day4.csv", day_index=3)
    build_day_csv(src, "data/day5.csv", day_index=4)
    build_day_csv(src, "data/day6.csv", day_index=5)

