import pandas as pd

class Building:
    """
    A Model of a building
    """
    def __init__(self):
        print("initializing Building object")
        # your code here...


    def load_params(self, path):
        """loads the sheet "params" of a excel at path and returns it as a dataframe"""

        # your code here...

        #return df

    def load_hull(self, path):
        """loads the sheet "thermal_ hull" of a excel at path and returns it as a dataframe"""

        # your code here...

        #return hull

    def insert_windows(self, hull_df, u_f, ff_anteil):
        """takes a hull dataframe from load_hull() and replaces an opak wall with a wall and a window entry, taking the window share and u-value as inputs"""

        # your code here...

        #return hull_df

    def L_T(self, hull_df):  # expects a pandas dataframe as input
        """calculates the LT from a Hull Dataframe"""

        # your code here...

        #return L_T

