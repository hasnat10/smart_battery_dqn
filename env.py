import numpy as np


class BatteryEnv:
    def __init__(
        self,
        load_profile,
        price_profile,
        capacity_kwh=13.5,
        max_charge_kw=5.0,
        max_discharge_kw=5.0,
        init_soc_fraction=0.5,
        dt_hours=1.0,
        use_forecast=False,
    ):

        if len(load_profile) != len(price_profile):
            raise ValueError("load and price must have same length")

        self.load = np.array(load_profile, dtype=float)
        self.price = np.array(price_profile, dtype=float)
        self.n_steps = len(self.load)

        self.capacity_kwh = float(capacity_kwh)
        self.max_charge_kw = float(max_charge_kw)
        self.max_discharge_kw = float(max_discharge_kw)
        self.init_soc_fraction = float(init_soc_fraction)
        self.dt_hours = float(dt_hours)
        self.use_forecast = bool(use_forecast)


        self.t = 0
        self.soc_kwh = 0.0
        self.total_cost = 0.0
        self.peak_import_kw = 0.0

        self.reset()

    def reset(self):
        self.t = 0
        self.soc_kwh = self.init_soc_fraction * self.capacity_kwh
        self.total_cost = 0.0
        self.peak_import_kw = 0.0
        return self._get_state()

    def _get_state(self):
        idx = min(self.t, self.n_steps - 1)
        if self.n_steps > 1:
            hour_norm = idx / (self.n_steps - 1)
        else:
            hour_norm = 0.0
        soc_frac = self.soc_kwh / self.capacity_kwh
        if self.use_forecast:
            start = max(0, idx - 2)
            hist = self.load[start : idx + 1]
            forecast = float(hist.mean())
            return np.array(
                [hour_norm, soc_frac, self.price[idx], self.load[idx], forecast],
                dtype=float,
            )
        return np.array(
            [hour_norm, soc_frac, self.price[idx], self.load[idx]],
            dtype=float,
        )


    def step(self, action):
        if action == 0:
            p_batt = self.max_charge_kw
        elif action == 1:
            p_batt = 0.0
        elif action == 2:
            p_batt = -self.max_discharge_kw
        else:
            raise ValueError("invalid action")

        if p_batt < 0.0:
            max_discharge = self.soc_kwh / self.dt_hours
            if -p_batt > max_discharge:
                p_batt = -max_discharge
        elif p_batt > 0.0:
            max_charge = (self.capacity_kwh - self.soc_kwh) / self.dt_hours
            if p_batt > max_charge:
                p_batt = max_charge

        new_soc = self.soc_kwh + p_batt * self.dt_hours
        if new_soc < 0.0:
            new_soc = 0.0
        if new_soc > self.capacity_kwh:
            new_soc = self.capacity_kwh

        load_kw = self.load[self.t]
        grid_import_kw = load_kw + p_batt
        if grid_import_kw < 0.0:
            grid_import_kw = 0.0

        price = self.price[self.t]
        step_cost = grid_import_kw * self.dt_hours * price

        self.total_cost += step_cost
        if grid_import_kw > self.peak_import_kw:
            self.peak_import_kw = grid_import_kw

        self.soc_kwh = new_soc
        self.t += 1

        done = self.t >= self.n_steps
        next_state = self._get_state()

        reward = -step_cost

        info = {
            "step_cost": step_cost,
            "grid_import_kw": grid_import_kw,
            "soc_kwh": self.soc_kwh,
            "total_cost": self.total_cost,
            "peak_import_kw": self.peak_import_kw,
        }

        return next_state, reward, done, info

