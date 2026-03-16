""" Plot the accuracy of BEM Green's function reconstruction as a function of minumum distance Rx_min.
The Rx_min is the minimum distance between the dipole and response point, also the x_target_min in the MATLAB script.
We showed this result in our paper. The alpha_value is manually created based on the data 
we got from verify_bem_fresnel.py for different Rx_min and wavelength. The plot is saved as calibration_distance.png.

Try to run this script:
python -m mqed.BEM.accuracy_plot

"""
import numpy as np 
from matplotlib import pyplot as plt

Rx_value = [1, 3, 5, 7, 10, 20, 30, 40, 50]
alpha_value_665 = [0.7969998834788722, 0.1434637476526375, 0.4884887575393157, 0.7259330785763067, 0.8631079451054886,0.9646911296158212, 0.9833886843889496, 0.9899108232274906,0.9929700363227106]
alpha_value_1000 = [0.7966710735932633, 0.14282505025770661, 0.4892551343263424, 0.7267586167516764, 0.8639694458463937, 0.9655617752706598, 0.984227517889597, 0.9906792786478612, 0.99362204940253]
alpha_value_300 = [0.7963787944447916, 0.14739647993760252, 0.4813750545022663, 0.7175825026046074, 0.8541344283411951, 0.9567863128941749, 0.9800532329399831, 0.9912513514980595, 0.9960682377717123]

alpha_value_combine = np.array([alpha_value_300, alpha_value_665, alpha_value_1000])
def plot_alpha_vs_Rx(Rx_value, alpha_value):
    """Plot alpha vs Rx for a given Rx value."""
    Rx_array = np.array(Rx_value)
    alpha_array = np.array(alpha_value)

    plt.figure(figsize=(8, 6))
    plt.plot(Rx_array, alpha_array[0], marker='o', linestyle='-', label='λ = 300 nm')
    plt.plot(Rx_array, alpha_array[1], marker='s', linestyle='--', label='λ = 665 nm')
    plt.plot(Rx_array, alpha_array[2], marker='^', linestyle='-.', label='λ = 1000 nm')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(Rx_array, fontsize=24, fontweight='bold')
    plt.yticks(np.linspace(0.2, 1.0, 5), fontsize=24, fontweight='bold')
    plt.xlabel('d (nm)', fontsize=24, fontweight='bold')
    plt.ylabel('$\\mathbf{\eta}$', fontsize=24, fontweight='bold')
    # plt.title('$\\alpha$ vs $Rx_{min}$', fontsize=16, fontweight='bold')
    # plt.grid(True, which="both", ls="--")
    leg = plt.legend(loc='lower right', fontsize=24, frameon=False)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')
    plt.tight_layout()
    # plt.yscale('log')
    plt.savefig('calibration_distance.png', dpi=400)
    plt.show()
    
plot_alpha_vs_Rx(Rx_value, alpha_value_combine)  