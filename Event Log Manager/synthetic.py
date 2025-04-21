import numpy as np
from datetime import timedelta, datetime
import random
import matplotlib.pyplot as plt
import pm4py
import energy_generator.ipynb

# Constants
MINUTES_IN_MONTH = 30 * 24 * 60  # minutes per month
SAMPLE_RATE = 60  # samples per second
SAMPLING_INTERVAL = 1  # minutes between samples

class DataGenerator:
    def __init__(self, startdate ,enddate, devices):

        self.startdate = startdate
        self.enddate = enddate
        self.interval = int((enddate - startdate).total_seconds() / SAMPLE_RATE)
        self.devices = devices
        self.powers = np.zeros(shape=(self.devices*2, self.interval))
        self.active_events = []


    def generate_data(self):

        generationtime = self.startdate
        generation_column = 0
        generation_frequency = timedelta(minutes=1)
        generation_events = pm4py.read_ocel_csv()

        while generationtime < self.enddate:
            device_nr = 0
            while device_nr < self.devices:
                power = self.power_generator(generationtime, generation_event)  # yield generation
                device_row_time = device_nr * 2
                device_row_power = device_row_time + 1
                self.powers[device_row_time, generation_column] = generationtime.timestamp()
                self.powers[device_row_power, generation_column] += power
                device_nr += 1

            generationtime += generation_frequency
            generation_column += 1

    def power_generator(self, generation_time):

        #get base cost from event type
        event_base_cost = get_event_base_from_time(generation_time)

        #add noise
        event_signal_cost = event_base_cost + random.uniform(10, 100)

        yield event_signal_cost

    def plot_data(self):
        for month in range(1, len(self.powers)):
            plt.plot([int(x / 1000) for x in np.arange(SAMPLING_INTERVAL, SAMPLING_INTERVAL + (len(self.powers[month]) * SAMPLING_INTERVAL), SAMPLING_INTERVAL)], self.powers[month], label=f"Month {month+1}")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Power (Watts)")
        plt.title(f"Synthetic Data for PC System Power Consumption")
        plt.legend()
        plt.show()

# Generate synthetic data
generator = DataGenerator(datetime.fromisocalendar(2018,1,1), datetime.fromisocalendar(2018, 52, 7), 1)
#data output should consist of a quadruple [a,b,c,d] where:
# a = complete signal that represent the machine
# b = time sensitive periodic component of the signal
# c = random noise component of the machine
# d = event induced signal component
generator.generate_data()
# Plot the data of the machine signal
generator.plot_data()