import numpy as np
import scipy.interpolate
import csv
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

def rand_unit_vect_2D():
    """Generate a unit 2-vector with random direction"""
    xy = np.random.normal(size=2)
    mag = sum(i**2 for i in xy) ** 0.5
    return xy / mag

def swim_speed_dist(num_particles, dist='swim_speed_distribution.csv'):
    """Produce a random swim speed for each particle based on the swim speed distribution for H. Akashiwo given in [Durham2013]"""

    # import the histogram (contains particle speed dist in um/s)
    with open(dist, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        bins = []
        counts = []
        for row in reader:
            bins.append(row[0])
            counts.append(row[1])

    # generate the PDF
    cum_counts = np.cumsum(counts)
    bin_width = 3
    x = cum_counts * bin_width
    y = bins
    pdf = scipy.interpolate.interp1d(x, y)
    b = np.zeros(num_particles)
    for i in range(len(b)):
        u = np.random.uniform(x[0], x[-1])
        b[i] = pdf(u)  # could convert in to meters if you want

    #plt.figure(figsize=(10, 6))
    #plt.hist(b, bins=100, density=True, alpha=1, color='black')
    #plt.xlabel('Swim Speed (um/s)')
    #plt.ylabel('PDF')
    #plt.xlim([0, np.max(b)])
    #plt.title('Swim Speed Distribution for H. Akashiwo')
    #plt.grid(True)

    #median_speed = np.median(b)
    # mean_speed = np.mean(b)
    #std_speed = np.std(b)
    
    #textbox_content = f'Median: {median_speed:.3f}\nMean: {mean_speed:.3f}\nStd: {std_speed:.3f}'
    #plt.text(0.98, 0.95, textbox_content, fontsize=13, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))
    #plt.savefig("LB_plankton_figs/PDF_Akashiwo.png",dpi=400)

    return b

def ini_swimspeed_cells(width, length, num_cells=1000):
    x0 = np.random.rand(num_cells,2) * np.array([width,length])
    p0 = np.array([rand_unit_vect_2D() for _ in range(num_cells)])

    vc = swim_speed_dist(num_cells)
    ini_velocities = np.zeros((num_cells,2))
    for i in range(num_cells):
        ini_velocities[i] = vc[i] * p0[i]

    return x0, vc, ini_velocities, p0

