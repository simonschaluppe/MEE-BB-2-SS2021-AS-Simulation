import pandas as pd

class Building:
    """
    A Model of a building
    """
    def __init__(self, path="data/building.xlsx", u_f=0.93, ff_anteil=0.4):

        self.file = path
        self.params_df      = self.load_params(path)
        self.bgf            = self.params_df.loc["gross_floor_area", "Value"]
        self.gf             = self.params_df.loc["plot_size", "Value"] # m²
        self.heat_capacity  = self.params_df.loc["effective_heat_capacity", "Value"] # Wh/m²K
        self.net_storey_height = self.params_df.loc["net_storey_height", "Value"]
        self.differential_cost = self.params_df.loc["differential_cost", "Value"]

        self.hull_df = self.load_hull(path)
        # if insert windows?
        self.hull_df = self.insert_windows(self.hull_df,
                                           u_f=u_f,
                                           ff_anteil=ff_anteil)

        self.LT = self.L_T(self.hull_df) / self.bgf # W/K /m²BGF

    def load_params(self, path):
        """loads the sheet "params" of a excel at path and returns it as a dataframe"""
        df = pd.read_excel(path, sheet_name="params")
        req_cols = {'Unit', 'Value', 'Variable'}
        common = req_cols.intersection(df.columns)
        if common != req_cols:
            raise ValueError(f"{path} sheet params is missing atleast one column names: {req_cols}")
        df.index = df["Variable"]
        return df

    def load_hull(self, path):
        """loads the sheet "thermal_ hull" of a excel at path and returns it as a dataframe"""
        hull = pd.read_excel(path, sheet_name="thermal_hull")
        return hull

    def insert_windows(self, hull_df, u_f, ff_anteil):
        """takes a hull dataframe from load_hull() and replaces an opak wall with a wall and a window entry, taking the window share and u-value as inputs"""
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
        """calculates the LT from a Hull Dataframe"""
        A_B = hull_df["Fläche"].sum()
        hull_df["L_B"] = hull_df["Fläche"] * hull_df["U-Wert"] * hull_df["Temperatur-Korrekturfaktor"]
        L_B = hull_df.L_B.sum()
        L_PX = max(0, (0.2 * (0.75 - L_B / A_B) * L_B))
        L_T = L_B + L_PX
        return L_T

    def __repr__(self):
        width = len(self.file) + 10
        string = f"""Building {self.file}
{"-" * width}
bgf: {self.gf}
gf: {self.heat_capacity}
heat_capacity: {self.net_storey_height}
net_storey_height: {self.differential_cost}
LT: {self.LT}
"""
        return string


if __name__ == "__main__":
    test = Building()