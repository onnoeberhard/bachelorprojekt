import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

# Configuration
sns.set()
plt.rcParams.update({
    'figure.figsize': (4*1.8, 3*1.8),
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'text.latex.preamble': [
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{siunitx}',
        r'\usepackage{physics}',
        r'\usepackage{amsmath}'
    ]
})

## Resistor and operating points
# Load spice data
ledvi = pd.read_csv('led_v_i.txt', '\t').values
afk_on = pd.read_csv('fet_id_uds_ugs5.txt', '\t').values
afk_off = pd.read_csv('fet_id_uds_ugs0.txt', '\t').values
operation = pd.read_csv('operation.txt', '\t').values

# Compute "off" operating point
led_v = interp1d(ledvi[:, 0], ledvi[:, 1])
fet_id = interp1d(afk_off[:, 0], afk_off[:, 1])
fet_v = 5
i_d = 0
points = np.zeros((4, 2))
print("Arbeitspunkt-Iteration:")
for i in range(4):
    print(f"{i_d:.5e} {5 - fet_v:.5e} {fet_v:.5e}")
    points[i, :] = np.r_[[fet_v, i_d]]
    i_d = fet_id(fet_v)
    fet_v = 5 - led_v(i_d)

# Plot LED V-I diagram and operating points
fig, ax = plt.subplots(constrained_layout=True)
ax.semilogx(ledvi[:, 0], ledvi[:, 1], label="U-I-Kennlinie der Diode (SPICE-Simulation)")
ax.plot([0.1], [1.495], 'go', label=r'Arbeitspunkt EIN $(\SI{100}{\mA}, \SI{1.495}{\V})$')
ax.plot(points[-1][1], 5 - points[-1][0], 'ro', label=r'Arbeitspunkt AUS $(\SI{34.4}{\nA}, \SI{0.879}{\V})$')
ax.set(xlabel='$I_D$ / A', ylabel='$U_D$ / V')
ax.set_xticks(10.**np.arange(-10, 1))
ax.xaxis.grid(True, which='minor', linewidth=.3)
ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10. ,subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=12))
ax.xaxis.set_major_formatter(EngFormatter(places=0, sep=""))
ax.legend()
fig.savefig('AP.png')

# Plot FET output curve and operating points / line
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(afk_on[:, 0], afk_on[:, 1]*1000, label=r"Ausgangskennlinie des FET bei $U_{GS} = \SI{5}{\V}$ (SPICE-Sim.)")
p = 100 + (100 - points[-1, 1])/(points[-1, 0] - 0.58)*0.58
ax.plot([0, points[-1, 0]], [p, points[-1, 1]], label='Arbeitsgerade')
ax.plot([0.58], [100], 'o', label=r'Arbeitspunkt EIN $(\SI{0.58}{\V}, \SI{100}{\mA})$')
ax.plot(points[-1, 0], points[-1, 1], 'o', label=r'Arbeitspunkt AUS $(\SI{4.12}{\V}, \SI{34.4}{\nA})$')
ax.set_xticks(list(range(6)) + [0.58])
ax.set(xlabel='$U_{DS}$ / V', ylabel='$I_D$ / mA')
ax.legend(loc='upper left')
fig.savefig('AKF.png')

# Plot switching characteristic 
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(operation[:, 0], operation[:, 1]*1000, label="SPICE-Simulation")
ax.plot([5], [90.67], 'go', label='Arbeitspunkt EIN')
ax.plot([0], [0], 'ro', label='Arbeitspunkt AUS')
ax.set_yticks(list(range(0, 90, 20)) + [90.67])
ax.set(xlabel='$U_{GS}$ / V', ylabel='$I_D$ / mA')
ax.legend()
fig.savefig('sim.png')

# Plot iteration
fig, ax = plt.subplots(1, 2, constrained_layout=True)
ax[0].semilogy(5 - ledvi[:, 1], ledvi[:, 0], label="$I_{D}$ (LED)")
ax[0].semilogy(afk_off[:, 0], afk_off[:, 1], "C2", label="$I_D$ (FET)")
ax[0].step(points[:, 0], points[:, 1], 'C3', label="Iteration")
ax[0].set_xticks(np.arange(0, 6))
ax[0].set_yticks(10.**np.arange(-10, 1))
ax[0].yaxis.grid(True, which='minor', linewidth=.3)
ax[0].yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10. ,subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=12))
ax[0].yaxis.set_major_formatter(EngFormatter(unit="A", places=0))
ax[0].set_xlabel("$U_{DS} = \SI{5}{\V} - U_{\mathrm{LED}}$")
ax[0].legend(loc='upper left')
ax[1].plot(5 - ledvi[:len(ledvi)//2, 1], ledvi[:len(ledvi)//2, 0], label="$I_{D}$ (LED)")
ax[1].plot(afk_off[len(afk_off)//2:, 0], afk_off[len(afk_off)//2:, 1], "C2", label="$I_D$ (FET)")
ax[1].step(points[:, 0], points[:, 1], 'C3', label="Iteration")
ax[1].yaxis.set_major_formatter(EngFormatter(unit="A", places=0))
ax[1].set_xlim(4.11, 4.13)
ax[1].set_ylim(3e-8, 4.5e-8)
ax[1].set_xlabel("$U_{DS} = \SI{5}{\V} - U_{\mathrm{LED}}$")
ax[1].legend(loc='upper right')
fig.savefig('iteration.png')

## Reference measurement
# Import and process MIREX data
reference = pd.read_csv('nephelometer_002.dat', '\t|;', index_col=0, skiprows=19, engine='python')
reference = reference.apply(lambda c: c.str.replace(',', '.')).astype(float).rename(columns=lambda c: c.strip())
reference.index = pd.to_datetime(reference.index) - pd.to_datetime(reference.index[0])
reference = reference.resample('500ms').interpolate('time')
reference.index.name = None
reference['sec'] = reference.index.values.astype(float) * 1e-9

# Import and process nephelometer data
nephelometer = pd.read_csv('messung.txt', '\ |,', index_col=0, engine='python').iloc[:, 1:]    # Import data
nephelometer.index = (pd.to_datetime(nephelometer.index)          # Normalize timestamps
                      - pd.to_datetime(nephelometer.index[0]))
nephelometer = nephelometer.resample('500ms').ffill()             # Upsample for synchronous data
nephelometer.index.name = None
nephelometer['sec'] = nephelometer.index.values.astype(float) * 1e-9

# The relation between integration times is not precisely linear (due to arduino program)
def scale(data_from, data_to):
    return LinearRegression(fit_intercept=False).fit(data_from[:, np.newaxis], data_to[:, np.newaxis]).coef_[0][0]

# Remove background measurement from measurement with LED turned on and normalize values.
for i in ['200', '150', '100', '50', '20', '10']:
    nephelometer[f'i{i}'] = (nephelometer[f'on_{i}ms'] 
                             - (scale(nephelometer.on_100ms, nephelometer[f'on_{i}ms'])
                                * nephelometer.off_100ms.mean()))

# Make both the MIREX and nephelometer measurements the same length
reference = reference[:len(nephelometer)]
assert len(reference) == len(nephelometer)

# Plot measurements
fig, ax = plt.subplots(1, 2, constrained_layout=True)
reference.plot(x='sec', y='MIREX', ax=ax[0])
ax[0].set_xlabel("Zeit [s]")
ax[0].set_ylabel("Dämpfung [dB]")
nephelometer.plot(x='sec', y=['on_200ms', 'on_150ms', 'on_100ms', 'on_50ms', 'on_20ms', 'on_10ms', 'off_100ms'], 
                  ax=ax[1])
ax[1].legend([r'\texttt{on\_200ms}', r'\texttt{on\_150ms}', r'\texttt{on\_100ms}', r'\texttt{on\_50ms}', 
              r'\texttt{on\_20ms}', r'\texttt{on\_10ms}', r'\texttt{off\_100ms}'])
ax[1].set_xlabel("Zeit [s]")
ax[1].set_ylabel("Messwert [0-1023]")
fig.savefig('messung.png')

# Plot shifted measurements
fig, ax = plt.subplots(2, 3, sharex=True, constrained_layout=True)
# Lighter colors
c = np.zeros((6, 3))
for i in range(6):
    x = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(f"C{i}"))
    x[1] **= 3
    x[2] **= 1/3
    c[i, :] = mpl.colors.hsv_to_rgb(x)

nephelometer.plot(x='sec', y='on_200ms', c=c[0, :], style='.', ax=ax[0, 0], 
                  title=r"$T = \SI{200}{\ms}$", label="Original", alpha=1)
nephelometer.plot(x='sec', y='on_150ms', c=c[1, :], style='.', ax=ax[0, 1],
                  title=r"$T = \SI{150}{\ms}$", label="Original", alpha=1)
nephelometer.plot(x='sec', y='on_100ms', c=c[2, :], style='.', ax=ax[0, 2],
                  title=r"$T = \SI{100}{\ms}$", label="Original", alpha=1)
nephelometer.plot(x='sec', y='on_50ms', c=c[3, :], style='.', ax=ax[1, 0],
                  title=r"$T = \SI{50}{\ms}$", label="Original", alpha=1)
nephelometer.plot(x='sec', y='on_20ms', c=c[4, :], style='.', ax=ax[1, 1],
                  title=r"$T = \SI{20}{\ms}$", label="Original", alpha=1)
nephelometer.plot(x='sec', y='on_10ms', c=c[5, :], style='.', ax=ax[1, 2],
                  title=r"$T = \SI{10}{\ms}$", label="Original", alpha=1)
nephelometer.plot(x='sec', y='i200', style='C0.', ax=ax[0, 0], title=r"$T = \SI{200}{\ms}$", label="Verschoben")
nephelometer.plot(x='sec', y='i150', style='C1.', ax=ax[0, 1], title=r"$T = \SI{150}{\ms}$", label="Verschoben")
nephelometer.plot(x='sec', y='i100', style='C2.', ax=ax[0, 2], title=r"$T = \SI{100}{\ms}$", label="Verschoben")
nephelometer.plot(x='sec', y='i50', style='C3.', ax=ax[1, 0], title=r"$T = \SI{50}{\ms}$", label="Verschoben")
nephelometer.plot(x='sec', y='i20', style='C4.', ax=ax[1, 1], title=r"$T = \SI{20}{\ms}$", label="Verschoben")
nephelometer.plot(x='sec', y='i10', style='C5.', ax=ax[1, 2], title=r"$T = \SI{10}{\ms}$", label="Verschoben")
ax[0, 0].set_ylabel("Messwert [0-1023]")
ax[1, 0].set_ylabel("Messwert [0-1023]")
for a in ax.flat:
    a.yaxis.grid(True, 'minor', linewidth=.3)
    a.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    a.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    a.set_xlabel("Zeit [s]")
    a.legend(fontsize=8, loc='upper left')

fig.savefig('verschiebung.png')

# Window function
def window(T=40, alpha=1):
    w = np.r_[np.linspace(0, 1, T)**alpha, T*[0]]
    return w / w.sum()    # Normalization

assert np.allclose([window(alpha=a).sum() for a in [.2, 1, 5]], [1]*3)    # Test normalization

# Smoothing function
def conv(data, T=40, alpha=1, pad=True):
    win = window(T, alpha)                                                 # Window function
    res = np.r_[T*[np.nan], convolve(data, win[::-1], mode='same')[T:]]    # Convolution
    if pad:
        res = np.r_[T*[data[:T].mean()], res[T:]]                          # Pad beginning
    return res

# Plot window functions and smoothed measurements
fig, ax = plt.subplots(1, 2, constrained_layout=True)
ax[0].plot(np.linspace(-20, 20, 80), window(alpha=5), 'C1', label=r'$\alpha = 5$')       # almost f (dirac)
ax[0].plot(np.linspace(-20, 20, 80), window(alpha=1), 'C2', label=r'$\alpha = 1$')       # linear
ax[0].plot(np.linspace(-20, 20, 80), window(alpha=.2), 'C3', label=r'$\alpha = 0.2$')    # almost mean (boxcar)
ax[0].set_xlabel(r"Zeit [s]")
ax[0].legend()
ax[1].plot(nephelometer.sec, nephelometer.i200.values, '.', alpha=.5, label='Messwerte ($T = \SI{200}{\ms}$)')
ax[1].plot(nephelometer.sec, conv(nephelometer.i200.values, alpha=5))
ax[1].plot(nephelometer.sec, conv(nephelometer.i200.values, alpha=1))
ax[1].plot(nephelometer.sec, conv(nephelometer.i200.values, alpha=.2))
ax[1].set_xlabel(r"Zeit [s]")
ax[1].legend()
fig.savefig('glatt.png')

# Find parameters using the 200ms curve to get the cleanest fit.
neph = conv(nephelometer.i200.values)
ref = reference['MIREX'].values

# Optimization to find actual parameters
t = np.arange(len(neph))
f_ref = interp1d(t, ref, bounds_error=False)
f_neph = interp1d(t, neph, bounds_error=False)

f_model = lambda t, a, b, c: c*f_neph(t - a) + b

def error(x):
    E = (f_ref(t) - f_model(t, x[0], x[1], x[2])) ** 2    # Least squares error
    E[:200] *= 2                                          # Especially penalize bias at 0
    return E[~np.isnan(E)].sum()

# Find optimal parameters using `error` as the objective function
x = None
for a in range(-50, 50):
    x_ = *minimize(error, np.r_[a, 0, 1], bounds=((-100, 100), (None, None), (None, None))).x,
    if x is None or error(x_) < error(x):
        x = x_

a, b, c = *x,    # Optimal parameters
print("\nGefundene Parameter:")
print(a, b, c)

# Plot fit with found parameters
fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True)
ax[0].plot(t/2, ref, label='MIREX')
ax[0].legend()
ax[0].set_ylabel("Dämpfung [dB]")
ax[1].plot(t/2, neph, label='Messwerte ($T = \SI{200}{\ms}$, geglättet)')
ax[1].legend()
ax[1].set_ylabel("Messwert [0-1023]")
ax[2].plot(t/2, f_ref(t), label='MIREX')
ax[2].plot(t/2, f_model(t, a, b, c), label='Messwerte (Verschoben und Skaliert)')
ax[2].legend()
ax[2].set_ylabel("Dämpfung [dB]")
ax[2].set_xlabel("Zeit [s]")
fig.savefig('optim.png')

# Rescale parameters for T = 100ms
k  = c*scale(nephelometer.i100.values, nephelometer.i200.values)
b  = b
t0 = a
print(f"\nParameters for 100ms integration:\nScale factor: k = {k}\nOffset: b = {b}\nTime delay: t0 = {t0}"
      "\nf(t) [dB] = k*m(t - t_0) + b")

# Test model on all measurements
def model(data):
    f = interp1d(t, conv(scale(data, nephelometer.i200.values)*data), bounds_error=False)
    return lambda t, a, b, c: c*f(t - a) + b

# Plot results
fig, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True)
ax[0].plot(t, f_ref(t), label='MIREX')
ax[0].plot(t, model(nephelometer.i200.values)(t, a, b, c), alpha=.3, label=r'$\SI{200}{ms}$')
ax[0].plot(t, model(nephelometer.i150.values)(t, a, b, c), alpha=.3, label=r'$\SI{150}{ms}$')
ax[0].plot(t, model(nephelometer.i100.values)(t, a, b, c), alpha=.3, label=r'$\SI{100}{ms}$')
ax[0].plot(t, model(nephelometer.i50.values)(t, a, b, c), alpha=.3, label=r'$\SI{50}{ms}$')
ax[0].plot(t, model(nephelometer.i20.values)(t, a, b, c), alpha=.3, label=r'$\SI{20}{ms}$')
ax[0].plot(t, model(nephelometer.i10.values)(t, a, b, c), alpha=.3, label=r'$\SI{10}{ms}$')
ax[0].legend()
ax[1].plot(t, f_ref(t), label='MIREX')
ax[1].plot(k*nephelometer.i100.values[int(-t0):] + b, '.',                   # Result without smoothing
           alpha=.5, label=r'Messwerte ($T = \SI{100}{\ms}$)')
ax[1].plot(k*interp1d(t, conv(nephelometer.i100.values),                     # Result with smoothing
           bounds_error=False)(t - t0) + b, label='Geglättete Messwerte')
ax[1].legend()
ax[0].set_ylabel("Dämpfung [dB]")
ax[0].set_xlabel("Zeit [s]")
ax[1].set_xlabel("Zeit [s]")
fig.savefig('final.png')

# Scale factors (lambda-factors)
sizes = [200, 150, 100, 50, 20, 10]
scales = np.full((6, 6), np.nan)
scales2 = np.full((6, 6), np.nan)
for i, a_ in enumerate(sizes):
    for j, b_ in enumerate(sizes[i:],  i):
        scales[i, j] = scale(nephelometer[f"i{a_}"], nephelometer[f"i{b_}"])
        scales2[i, j] = scales[i, j] / (sizes[j] / sizes[i])

# Plot scale factors
fig, ax = plt.subplots(2, 2, sharex='col', constrained_layout=True)
ax[0, 1].plot(sizes, np.r_[sizes] / 200, 'C1-', label=r"Ideal ($\frac{T_2}{T_1}$)")
ax[0, 1].plot(sizes, scales[0, :], 'C0.-', label="Tatsächlich")
ax[0, 1].text(120, 0.25, r"$T_1 = \SI{200}{\ms}$")
ax[0, 1].legend()
ax[0, 1].set_ylabel(r"$\lambda_{T_1\rightarrow T_2}$")
ax[1, 1].plot(sizes, len(sizes)*[1], 'C1-', label="Ideal (1)")
ax[1, 1].plot(sizes, scales2[0, :], 'C0.-', label="Tatsächlich")
ax[1, 1].text(120, 1.06, r"$T_1 = \SI{200}{\ms}$")
ax[1, 1].legend()
ax[1, 1].set_ylabel(r"$\lambda_{T_1\rightarrow T_2}\cdot\frac{T_1}{T_2}$")
ax[1, 1].set_xticks(sizes)
ax[1, 1].set_xlabel(r"$T_2$ [ms]")
im = ax[0, 0].imshow(scales, cmap='viridis', vmin=ax[0, 1].get_ylim()[0], vmax=ax[0, 1].get_ylim()[1])
cbar = fig.colorbar(im, ax=ax[0, 0])
ax[0, 0].grid(False)
ax[0, 0].set_xticks(np.arange(0, 6))
ax[0, 0].yaxis.set_major_formatter(mpl.ticker.IndexFormatter(sizes))
ax[0, 0].set_ylabel(r"$T_1$ [ms]")
im = ax[1, 0].imshow(scales2, cmap='viridis', vmin=ax[1, 1].get_ylim()[0], vmax=ax[1, 1].get_ylim()[1])
cbar = fig.colorbar(im, ax=ax[1, 0])
ax[1, 0].grid(False)
ax[1, 0].set_xticks(np.arange(0, 6))
ax[1, 0].yaxis.set_major_formatter(mpl.ticker.IndexFormatter(sizes))
ax[1, 0].xaxis.set_major_formatter(mpl.ticker.IndexFormatter(sizes))
ax[1, 0].set_xlabel(r"$T_2$ [ms]")
ax[1, 0].set_ylabel(r"$T_1$ [ms]")
fig.savefig('linear.png')

# Export 100ms data for error checking in matlab
export = nephelometer[['off_100ms', 'on_100ms']]
export.columns = ['off', 'on']
export.to_csv('reference_100ms.csv')
