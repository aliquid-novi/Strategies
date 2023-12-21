class MonteCarloSimulation:
    def __init__(self, mean, std, periods, simulations, initial_price):
        self.mean = mean
        self.std = std
        self.periods = periods
        self.simulations = simulations
        self.initial_price = initial_price
        self.simulation_results = None

    def run_simulation(self):
        results = np.zeros((self.periods, self.simulations))
        results[0] = self.initial_price

        for t in range(1, self.periods):
            random_shock = np.random.normal(self.mean, self.std, size=self.simulations)
            results[t] = results[t-1] * np.exp(random_shock)

        self.simulation_results = results

    def plot_results(self):
        if self.simulation_results is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(pd.DataFrame(self.simulation_results))
            plt.title("Monte Carlo Simulation")
            plt.xlabel("Time Periods")
            plt.ylabel("Price")
            plt.show()
        else:
            print("Run the simulation first.")
    
    def test(self):
        print('test')
