# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:10:17 2021

@author: Simon
"""

import numpy as np


class PV:
    """
    PV Profile with .TSD in [kWh per hour]
    """

    def __init__(self, csv="data/pv_1kWp.csv", kWp=1, cost_kWp=1500):
        default_path = "data/pv_1kWp.csv"

        if csv == default_path:
            print(f"No csv-path given, loading from default {default_path}...")
            self.TSD = np.genfromtxt(csv)

        else:
            self.TSD = np.genfromtxt(csv)  # no specifiers, better make sure that this is right
        self.path = csv
        self.kWp = kWp
        self.cost_kWp = cost_kWp # cost per kWh
        self.cost = kWp * self.cost_kWp

    def set_kWp(self, kWp):
        self.TSD = self.TSD / self.kWp * kWp
        self.kWp = kWp
        self.cost = kWp * self.cost_kWp

    def __repr__(self):
        width = len(self.path)+10
        return f"""PV-System {self.path}
{"-"*width}
kWp: {self.kWp:>{width-5}d}
kWh/a: {self.TSD.sum():>{width-7}.0f}
cost [€]: {self.cost:>{width-10}d}"""


if __name__ == "__main__":
    test = PV()
    import matplotlib.pyplot as plt
    plt.plot(test.TSD)
    plt.show()
