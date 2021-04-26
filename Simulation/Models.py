import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_Battery = {
    "name":     "test Battery",
    "cost":     400., #€/kWh
    "size":     100., #kWh Capacity, defined boundaries
    "losses":   0.03, #kWh/h
    "charging_power": 10., #kW
    "discharging_power": 10. #kW
}

from PV import pv

class Building:
    """
    A thermal Model of a building
    """
    def __init__(self, path="data/building.xlsx"):

        self.params_df      = self.load_params(path)
        self.bgf            = self.params_df.loc["gross_floor_area", "Value"]
        self.gf             = self.params_df.loc["plot_size", "Value"] # m²
        self.heat_capacity  = self.params_df.loc["effective_heat_capacity", "Value"] # Wh/m²K
        self.net_storey_height = self.params_df.loc["net_storey_height", "Value"]
        self.differential_cost = self.params_df.loc["differential_cost", "Value"]

        self.hull_df = self.load_hull(path)
        # if insert windows?
        self.hull_df = self.insert_windows(self.hull_df,
                                           u_f=1.5,
                                           ff_anteil=0.4)

        self.LT = self.L_T(self.hull_df) / self.bgf # W/K /m²BGF

    def load_params(self, path):
        df = pd.read_excel(path, sheet_name="params")
        req_cols = {'Unit', 'Value', 'Variable'}
        common = req_cols.intersection(df.columns)
        if common != req_cols:
            raise ValueError(f"{path} sheet params is missing atleast one column names: {req_cols}")
        df.index = df["Variable"]
        return df

    def load_hull(self, path):
        hull = pd.read_excel(path, sheet_name="thermal_hull")
        return hull

    def insert_windows(self, hull_df, u_f, ff_anteil):
        aw_A = hull_df.loc[0, "Fläche"] # funktioniert, aber nur das erste mal. sollte stattdessen aktiv eine "Außenwand (brutto)" suchen, wenns die nicht gibt macht der restliche code nicht viel sinn
        aw_opak_A = aw_A * (1 - ff_anteil)
        fenster_A = aw_A * ff_anteil

        aw_opak = dict(zip(hull_df.columns, ["AW (opak)", aw_opak_A, hull_df.loc[0, "U-Wert"],
                                             hull_df.loc[0, "Temperatur-Korrekturfaktor"]]))
        fenster = dict(zip(hull_df.columns, ["Fenster", fenster_A, u_f, hull_df.loc[0, "Temperatur-Korrekturfaktor"]]))

        hull_df = hull_df.append(aw_opak, ignore_index=True)
        hull_df = hull_df.append(fenster, ignore_index=True)
        hull_df.drop(hull_df.index[0], inplace=True)
        return hull_df

    def L_T(self, hull_df):  # expects a pandas dataframe as input
        A_B = hull_df["Fläche"].sum()
        hull_df["L_B"] = hull_df["Fläche"] * hull_df["U-Wert"] * hull_df["Temperatur-Korrekturfaktor"]
        L_B = hull_df.L_B.sum()
        L_PX = max(0, (0.2 * (0.75 - L_B / A_B) * L_B))
        L_T = L_B + L_PX
        return L_T

class Battery:
    """a Simplistic Battery"""
    def __init__(self, kWh):
        self.capacity = kWh # kWh
        self.charge_power_max = 10 # kW
        self.discharge_power_max = 10 # kW
        self.charge_efficiency = 0.9
        self.discharge_efficiency = 0.9
        self.discharge_per_hour = 0.00012
        self.cost_kWh = 1000
        self.cost = self.capacity * self.cost_kWh
        self.current_charge = 0. #kWh

    def charge(self, kW):
        max_charge = (self.capacity - self.current_charge) / self.charge_efficiency
        accepted_energy = min(kW, self.charge_power_max, max_charge)
        self.current_charge += accepted_energy * self.charge_efficiency
        return accepted_energy

    def discharge(self, kW:float):
        max_discharge = min(self.discharge_power_max, self.current_charge)
        desired_discharge = kW / self.discharge_efficiency
        discharged_energy = min(desired_discharge, max_discharge)
        self.current_charge -= discharged_energy
        return discharged_energy * self.discharge_efficiency

class Config:
    """stuff that won't change for most simulations"""
    heating_system = True
    heating_eff = 0.95 # Wirkungsgrad Verteilverluste
    heating_months = [1,2,3,4,9,10,11,12]
    minimum_room_temperature = 20.

    HP_COP = 5 #
    HP_heating_power = 20 #W/m²

    cooling_system = True
    cooling_months = [4,5,6,7,8,9]
    maximum_room_temperature = 26.

    DHW_COP = 3

    cp_air = 0.34 # spez. Wärme kapazität Luft (Wh/m3K)

    price_grid = 0.19
    price_feedin = 0.05


class Model:
    def __init__(self, kWp=1, battery_kWh=15):

        self.building = Building()
        self.config = Config()

        self.PV = pv(kWp=kWp)
        self.PV_prod = self.PV.TSD *1000 / self.building.bgf  # everything is in Wh/m²
        self.PV_use = np.zeros(8760)
        self.PV_feedin = np.zeros(8760)
        self.PV_to_battery = np.zeros(8760)


        self.battery = Battery(kWh=battery_kWh)
        self.Btt_to_ED = np.zeros(8760)

        # load usage characteristics
        Usage = pd.read_csv("data/usage_profiles.csv", encoding="cp1252")
        self.QI_winter = Usage["Qi Winter W/m²"].to_numpy()
        self.QI_summer = Usage["Qi Sommer W/m²"].to_numpy()
        self.QI = self.QI_winter
        self.ACH_V = Usage["Luftwechsel_Anlage_1_h"].to_numpy()
        self.ACH_I = Usage["Luftwechsel_Infiltration_1_h"].to_numpy()
        self.Qdhw = Usage["Warmwasserbedarf_W_m2"].to_numpy()
        self.ED_user = Usage["Nutzerstrom_W_m2"].to_numpy()

        # load climate data
        self.TA = np.genfromtxt("data/climate.csv",
                                delimiter=";")[1:, 1]
        # load solar gains
        self.QS = np.genfromtxt("data/Solar_gains.csv") # W/m²

        # initialize result arrays
        self.timestamp = pd.Series(np.arange('1970-01-01 00:00', '1971-01-01 00:00', dtype='datetime64[h]'))

        self.QV = np.zeros(8760) * np.nan # ventilation losses
        # it's usually better to initialize our results with Not A Number,
        # as it will immediatly cry if something isnt working.
        # Zeros can silence errors and we never want that

        self.QT = np.zeros(8760) # transmission losses
        self.TI = np.zeros(8760) # indoor temperature

        self.QH = np.zeros(8760)  * np.nan # Heating demand Wh/m²
        self.QC =  np.zeros(8760) * np.nan # Cooling demand Wh/m²
        self.Q_loss = np.zeros(8760) * np.nan

        self.ED_QH = np.zeros(8760) * np.nan # Electricity demand for heating Wh/m²
        self.ED_QC = np.zeros(8760) * np.nan  # Electricity demand for cooling Wh/m²

        #self.ED_Qdhw = 0

        self.ED = np.zeros(8760) * np.nan # Electricity demand Wh/m²



        self.ED_grid = np.zeros(8760)

    def init_sim(self):
        self.QV[0] = 0
        self.QT[0] = 0
        self.TI[0] = self.config.minimum_room_temperature
        self.QH[0] = 0
        self.QC[0] = 0
        self.Q_loss[0] = 0
        self.ED_QH[0] = 0
        self.ED_QC[0] = 0
        self.ED[0] = 0

    def calc_QV(self, t):
        """Ventilation heat losses [W/m²NGF] at timestep t"""
        dT = self.TA[t - 1] - self.TI[t - 1]
        room_height = self.building.net_storey_height
        cp_air = self.config.cp_air
        # thermally effective air change
        eff_airchange = self.ACH_I[t] + self.ACH_V[t]  # * M.VentilationSystem.share_cs * rel_ACH_after_heat_recovery

        self.QV[t] = eff_airchange * room_height * cp_air * dT

    def calc_QT(self, t):
        """Transmission heat losses [W/m²NGF] at timestep t"""
        dT = self.TA[t - 1] - self.TI[t - 1]
        self.QT[t] = self.building.LT * dT

    def is_heating_on(self, t, TI_new):
        if self.config.heating_system == True:
            if self.timestamp[t].month in self.config.heating_months:
                if TI_new < self.config.minimum_room_temperature:
                    return True
        return False

    def calc_QH(self, t):
        """Heating demand"""
        self.QH[t] = 0
        self.Q_loss[t] = (self.QT[t] + self.QV[t]) + self.QS[t] + self.QI[t]
        TI_new = self.TI[t - 1] + self.Q_loss[t] / self.building.heat_capacity

        if self.is_heating_on(t, TI_new):
            self.QH[t] = (self.config.minimum_room_temperature - TI_new) * self.building.heat_capacity

    def calc_ED_QH(self, t):
        """Calculates the necessary electricity demand"""
        required_for_heating = self.QH[t] / self.config.HP_COP / self.config.heating_eff
        available_power = self.config.HP_heating_power

        self.ED_QH[t] = min(required_for_heating, available_power)
        # calc_DHW(M, TSD, t)

    def is_cooling_on(self, t, TI_new):
        """
        Determines, whether all conditions are met to use cooling
        """
        c1 = self.config.cooling_system == True
        c2 = self.timestamp[t].month in self.config.cooling_months
        c3 = TI_new > self.config.maximum_room_temperature
        return all([c1, c2,
                    c3])  # returns True if all conditions are true, False otherwise. similarly, any(). You can stack this way more cleanly

    def calc_QC(self, t):
        self.QC[t] = 0
        self.Q_loss[t] = (self.QT[t] + self.QV[t]) + self.QS[t] + self.QI[t]
        TI_new = self.TI[t - 1] + self.Q_loss[t] / self.building.heat_capacity

        if self.is_cooling_on(t, TI_new):
            self.QC[t] = (self.config.maximum_room_temperature - TI_new) * self.building.heat_capacity

    def calc_ED_QC(self, t):
        """Calculates the necessary electricity demand for cooling"""
        required_for_cooling = - self.QC[t] / self.config.HP_COP / self.config.heating_eff
        available_power = self.config.HP_heating_power

        self.ED_QC[t] = min(required_for_cooling, available_power)
        # calc_DHW(M, TSD, t)

    def calc_ED(self, t):
        self.ED[t] = self.ED_QH[t] + self.ED_QC[t] + self.ED_user[t]

    def calc_PV_use(self, t):
        """allocates the PV to use"""
        self.PV_use[t] = min(self.PV_prod[t], self.ED[t])
        remain = self.PV_prod[t] - self.PV_use[t]

        self.PV_to_battery[t] = self.battery.charge(remain * self.building.bgf / 1000) * 1000 / self.building.bgf
        remain = remain - self.PV_to_battery[t]

        self.PV_feedin[t] = max(remain - self.ED[t], 0)

    def calc_battery_use(self,t):
        remaining_ED = (self.ED[t] - self.PV_use[t]) * self.building.bgf / 1000 #kW not W/m²
        # conditions
        c1 = (remaining_ED > 0)
        c2 = (self.battery.current_charge > 0)
        if all([c1, c2]):
            self.Btt_to_ED[t] = self.battery.discharge(remaining_ED)

    def calc_TI(self, t):
        QH_effective = self.ED_QH[t] * self.config.HP_COP * self.config.heating_eff
        QC_effective = -self.ED_QC[t] * self.config.HP_COP * self.config.heating_eff

        Q_balance = QH_effective + QC_effective + (self.QT[t] + self.QV[t]) + self.QS[t] + self.QI[t]
        self.TI[t] = self.TI[t - 1] + Q_balance / self.building.heat_capacity

    def calc_cost(self, years=20):
        """calculates the total cost of the system"""
        # calc investment
        self.investment_cost = self.building.differential_cost * self.building.bgf + self.PV.cost + self.battery.cost
        self.operational_cost = self.building.bgf * (
                                - self.PV_feedin.sum()/1000 * self.config.price_feedin \
                                + self.ED_grid.sum()/1000 * self.config.price_grid)

        print(f"Investment cost:  {round(self.investment_cost):>20.2f} €")
        print(f"Operational cost: {round(self.operational_cost):>20.2f} €/annum")

        self.total_cost = self.investment_cost + self.operational_cost * years
        print(f"Total cost after {years} years: {round(self.total_cost):>11,.2f} €")

        return self.total_cost

    def simulate(self):

        self.init_sim()

        for t in range(1, 8760):
            #### Verluste
            self.calc_QV(t)
            self.calc_QT(t)
            #### Heizung
            self.calc_QH(t)
            self.calc_ED_QH(t)
            #### Kühlung
            self.calc_QC(t)
            self.calc_ED_QC(t)
            #### Raumtemperatur
            self.calc_TI(t)

            #calc total energy demand
            self.calc_ED(t)
            #allocate pv
            self.calc_PV_use(t)
                # use directly
                # power battery
                # feed_in
            # discharge battery
            self.calc_battery_use(t)
            self.ED_grid[t] = self.ED[t] - self.PV_use[t] - self.Btt_to_ED[t]

        self.calc_cost()
        return True

    def plot(self, show=True, start=1, end=400):
        fig, ax = plt.subplots(2,2)#,figsize=(8,12)) #tight_layout=True)
        ax = ax.flatten()
        self.plot_Q(fig, ax[0], start=start, end=end)
        self.plot_T(fig, ax[1], start=start, end=end)
        self.plot_ED(fig, ax[2], start=start, end=end)
        self.plot_ES(fig, ax=ax[3], start=start, end=end)

        if show:
            dummy = plt.figure() # create a dummy figure
            new_manager = dummy.canvas.manager  # and use its manager to display "fig"
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)
            fig.show()

    def plot_Q(self, fig, ax, start=1, end=8760):
        # FigureCanvas(fig) # not needed in mpl >= 3.1

        ax.plot(self.QT[start:end])
        ax.plot(self.QV[start:end])
        ax.plot(self.QS[start:end])
        ax.plot(self.QI[start:end])
        ax.plot(self.QH[start:end])
        ax.plot(self.QC[start:end])
        ax.set_title("Wärmebilanz")
        ax.set_ylabel("W/m²")
        ax.legend(["Transmissionsverluste", "Lüftungsverluste", "Solare Gewinne",
                   "Innere Lasten", "Heizwärmebdedarf", "Kühlbedarf"])

    def plot_T(self, fig, ax, start=1, end=8760):
        # FigureCanvas(fig) # not needed in mpl >= 3.1
        ax.plot(self.TI[start:end])
        ax.plot(self.TA[start:end])
        ax.set_title("Temperatur")
        ax.set_ylabel("Temperatur [°C]")
        ax.legend(["Innenraum", "Außenluft"])

    def plot_ED(self, fig, ax, start=1, end=8760):
        # FigureCanvas(fig) # not needed in mpl >= 3.1
        ax.plot(self.PV_prod[start:end])
        ax.plot(self.ED_QH[start:end])
        ax.plot(self.ED_QC[start:end])
        ax.plot(self.ED_user[start:end])
        ax.set_title("Strom")
        ax.set_ylabel("W/m²")
        ax.legend(["PV", "WP Heizen", "WP Kühlen", "Nutzerstrom"])

    def plot_ES(self, fig=None, ax=None, start=1, end=8760):
        """plots the electricity supply and use"""
        show_self=False
        if not ax:
            fig, ax = plt.subplots(1,1)
            show_self = True
        ax.stackplot(range(start, end),
                     self.PV_use[start:end],
                     self.Btt_to_ED[start:end],
                     self.ED_grid[start:end],
                     self.PV_to_battery[start:end],
                     self.PV_feedin[start:end],)
        ax.set_title("PV Nutzung")
        ax.set_ylabel("W/m²")
        ax.legend(['PV Eigenverbrauch',
                   'Batterie-Entladung',
                   'Netzstrom',
                   'Batterie-Beladung',
                   'Einspeisung'])
        if show_self:
            fig.show()



if __name__ == "__main__":
    m = Model(kWp=25, battery_kWh=5)
    m.simulate()
    m.plot(start=4400,end=4600)
