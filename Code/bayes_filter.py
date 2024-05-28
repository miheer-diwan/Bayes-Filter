import numpy as np

class BayesFilter:
    def __init__(self):
        # Prior Probabilities of initial states:
        self.bel_0_open = 0.5  # belief that the door is initially open
        self.bel_0_closed = 0.5  # belief that the door is initially closed

        # Possible actions: u1 = do_nothing, u2 = open
        self.p_open_u1_open = 1
        self.p_closed_u1_open = 0
        self.p_open_u1_closed = 0
        self.p_closed_u1_closed = 1

        self.p_open_u2_open = 1
        self.p_closed_u2_open = 0
        self.p_open_u2_closed = 0.8
        self.p_closed_u2_closed = 0.2

        # Possible sensor values: z1 = open, z2 = closed
        self.p_z1_open = 0.6
        self.p_z1_closed = 0.2

        self.p_z2_open = 0.4
        self.p_z2_closed = 0.8

    def predict(self, action, bel_open, bel_closed):
        if action == 0:  # do_nothing
            bel_bar_open = self.p_open_u1_open * bel_open + self.p_open_u1_closed * bel_closed
            bel_bar_closed = self.p_closed_u1_open * bel_open + self.p_closed_u1_closed * bel_closed
        elif action == 1:  # open
            bel_bar_open = self.p_open_u2_open * bel_open + self.p_open_u2_closed * bel_closed
            bel_bar_closed = self.p_closed_u2_open * bel_open + self.p_closed_u2_closed * bel_closed
        return bel_bar_open, bel_bar_closed

    def measure(self, sensor_value, bel_bar_open, bel_bar_closed):
        if sensor_value == 1:  # open
            bel_open = self.p_z1_open * bel_bar_open
            bel_closed = self.p_z1_closed * bel_bar_closed
        elif sensor_value == 0:  # closed
            bel_open = self.p_z2_open * bel_bar_open
            bel_closed = self.p_z2_closed * bel_bar_closed
        eta = 1 / (bel_open + bel_closed)
        bel_open *= eta
        bel_closed *= eta
        return bel_open, bel_closed

    def bayes_filter(self, action, measurement, threshold):
        bel_open, bel_closed = self.bel_0_open, self.bel_0_closed
        iteration_count = 0
        while True:
            bel_bar_open, bel_bar_closed = self.predict(action, bel_open, bel_closed)
            bel_open, bel_closed = self.measure(measurement, bel_bar_open, bel_bar_closed)
            iteration_count += 1
            if bel_open >= threshold:
                break
        return bel_bar_open, bel_bar_closed, iteration_count
