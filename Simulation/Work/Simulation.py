# imports

# your code here ...

class Model:
    def __init__(self,building_path="data/building.xlsx",
                 kWp=0,
                 battery_kWh=0):
        print("initializing Model")
        ###### Compononets #####
        # (Other classes and parts, that form the model)
        # your code here...

        ###### Parameters #####
        # your code here...

        ###### Timeseries #####
        # load Usage characteristics
        # your code here...

        # load climate data
        # your code here...

        # load solar gains
        # your code here...

    # Define your methods to calculate and handle simulation data here
    # your code here...

    def init_sim(self):
        print("initializing simulation")
        # (re)load profiles from self.Usage dataframe
        #this is neccessary  if the PV model has changed inbetween simulations

        # your code here...

        # (re)load PV profiles
        # your code here...

        # initialize result arrays
        # your code here...

        ## initialize starting conditions on t=0
        # (like indoor temperature, storage SoC, ...)
        # your code here...

    def simulate(self):
        # your code here...
        for t in range(1, 8760):
            print("simulating timestep ",t)
            # your code here...
        # your code here...
        return True


if __name__ == "__main__":
    m = Model(kWp=25, battery_kWh=30)
    m.simulate()
