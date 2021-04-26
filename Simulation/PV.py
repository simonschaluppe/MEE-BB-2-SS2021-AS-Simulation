# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:10:17 2021

@author: Simon
"""

import numpy as np


class pv:
    """
    PV Profile with .TSD in [kWh/a]
    """

    def __init__(self, csv="data/pv_1kWp.csv", kWp=1, cost_kWp=1000):
        default_path = "data/pv_1kWp.csv"

        if csv == default_path:
            print(f"No csv-path given, loading from default {default_path}...")
            self.TSD = np.genfromtxt(csv)
        else:
            self.TSD = np.genfromtxt(csv)  # no specifiers, better make sure that this is right

        self.cost_kWp = cost_kWp
        self.set_kWp(kWp)

    def set_kWp(self, kWp):
        self.TSD = self.TSD * kWp
        self.kWp = kWp
        self.cost = kWp * self.cost_kWp


if __name__ == "__main__":
    test = pv()
    import matplotlib.pyplot as plt

    plt.plot(test.TSD)
