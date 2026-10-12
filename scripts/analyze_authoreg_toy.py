import csv
import math
import sys


writer = csv.writer(sys.stdout)
writer.writerow(("signal_scale", "noise_std", "optimal_gain", "mse", "mutual_information_nats"))
for noise_std in (0.1, 0.5, 1.0):
    for signal_scale in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        signal_power = signal_scale ** 2
        noise_power = noise_std ** 2
        gain = signal_power / (signal_power + noise_power)
        mse = signal_power * noise_power / (signal_power + noise_power)
        mutual_information = 0.5 * math.log1p(signal_power / noise_power)
        writer.writerow((signal_scale, noise_std, gain, mse, mutual_information))
