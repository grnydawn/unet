# target_script2.py

import numpy as np
import matplotlib.pyplot as plt

def plot_sine_wave():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    #@boda

    plt.figure()
    plt.plot(x, y, label='sin(x)')
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True)
    print("TARGET CODE")
    #plt.show()
