import matplotlib.pyplot as plt


def plot_raster(spike_data, time_range=None, title="Raster Plot"):
    trains = spike_data.train
    duration = spike_data.length / 1000.0
    t_start, t_end = time_range or (0, duration)

    plt.figure(figsize=(14, 5))
    for y, spikes in enumerate(trains):
        spikes = [s for s in spikes if t_start <= s <= t_end]
        plt.scatter(spikes, [y] * len(spikes), marker="|", color="black", s=4)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index")
    plt.xlim(t_start, t_end)
    plt.tight_layout()
    plt.show()


def plot_population_rate(times, rate, bursts=None, title="Population FR"):
    plt.figure(figsize=(14, 4))
    plt.plot(times, rate, label="Smoothed FR", color="red")

    if bursts:
        for b in bursts:
            plt.axvspan(b["start"] / 1000, b["end"] / 1000, color="gray", alpha=0.3)
            plt.axvline(b["t_peak"] / 1000, color="red", linestyle="--", lw=1)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
