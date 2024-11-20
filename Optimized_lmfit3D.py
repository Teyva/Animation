import os
import numpy as np
import lmfit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from PIL import Image
from scipy import stats, optimize

sns.set_style('whitegrid')

def load_tiff_sequence(tiff_path, bit_depth):
    """
    Charge une séquence TIFF dans un tableau numpy 3D.
    - tiff_path : chemin vers le fichier TIFF
    - bit_depth : profondeur de bits (8 ou 16)
    """
    img = Image.open(tiff_path)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frame = img.convert('L')  # Conversion en niveaux de gris
        frames.append(np.array(frame, dtype=np.float64))
    frames = np.array(frames)
    normalize_factor = 65535.0 if bit_depth == 16 else 255.0
    frames /= normalize_factor
    return frames, img.n_frames

def gaussian_2d(x, y, A, x0, y0, sigma_x, sigma_y, B):
    """
    Fonction Gaussienne 2D.
    - x, y : coordonnées
    - A : amplitude
    - x0, y0 : centre
    - sigma_x, sigma_y : écart-types
    - B : offset
    """
    return A * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    ) + B

def lmfit_process(Z):
    """
    Effectue un ajustement Gaussien sur des données 2D avec paramètres initiaux optimisés.
    - Z : image 2D à analyser
    Retourne les paramètres de l'ajustement.
    """
    height, width = Z.shape
    y, x = np.indices(Z.shape)
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = Z.ravel()

    def objective_function(param_values):
        A, x0, y0, sigma_x, sigma_y, B = param_values
        model_values = gaussian_2d(x_flat, y_flat, A, x0, y0, sigma_x, sigma_y, B)
        residual = z_flat - model_values
        return np.sum(residual ** 2)

    initial_guess = [
        Z.max() - Z.min(),    # A
        width / 2,            # x0
        height / 2,           # y0
        width / 4,            # sigma_x
        height / 4,           # sigma_y
        Z.min(),              # B
    ]

    bounds = (
        [0, 0, 0, 1e-5, 1e-5, -np.inf],        
        [np.inf, width - 1, height - 1, width, height, np.inf],
    )

    result_opt = optimize.least_squares(
        objective_function,
        x0=initial_guess,
        bounds=bounds,
        method='trf',
        max_nfev=100
    )

    optimized_params = {
        'A': result_opt.x[0],
        'x0': result_opt.x[1],
        'y0': result_opt.x[2],
        'sigma_x': result_opt.x[3],
        'sigma_y': result_opt.x[4],
        'B': result_opt.x[5],
    }

    model = lmfit.Model(gaussian_2d, independent_vars=['x', 'y'])
    params = model.make_params(**optimized_params)
    params['A'].min = 0
    params['sigma_x'].min = 1e-5
    params['sigma_y'].min = 1e-5
    params['x0'].set(min=0, max=width - 1)
    params['y0'].set(min=0, max=height - 1)
    params['B'].set(min=-np.inf, max=np.inf)

    result = model.fit(z_flat, params, x=x_flat, y=y_flat)

    sigma_x_value = result.params['sigma_x'].value
    sigma_y_value = result.params['sigma_y'].value
    sigma_x_error = result.params['sigma_x'].stderr or 0
    sigma_y_error = result.params['sigma_y'].stderr or 0

    mu_x_value = result.params['x0'].value
    mu_y_value = result.params['y0'].value
    mu_x_error = result.params['x0'].stderr or 0
    mu_y_error = result.params['y0'].stderr or 0

    radius_value = (sigma_x_value + sigma_y_value) / 2
    radius_error = np.sqrt((sigma_x_error ** 2 + sigma_y_error ** 2) / 4)

    return {
        "radius": radius_value,
        "radius_error": radius_error,
        "position": (mu_x_value, mu_y_value),
        "position_error": (mu_x_error, mu_y_error),
    }

def compute_trajectory(images):
    """
    Calcule la trajectoire, le MSD et les erreurs pour une séquence d'images.
    - images : séquence d'images 3D
    Retourne les positions, rayons et MSD.
    """
    n_frames = images.shape[0]
    radius = np.zeros(n_frames)
    radius_errors = np.zeros(n_frames)
    positions_x = np.zeros(n_frames)
    positions_y = np.zeros(n_frames)
    position_x_errors = np.zeros(n_frames)
    position_y_errors = np.zeros(n_frames)

    for i in range(n_frames):
        fit_results = lmfit_process(images[i])
        radius[i] = fit_results["radius"]
        radius_errors[i] = fit_results["radius_error"]
        positions_x[i] = fit_results["position"][0]
        positions_y[i] = fit_results["position"][1]
        position_x_errors[i] = fit_results["position_error"][0]
        position_y_errors[i] = fit_results["position_error"][1]

    dx = np.diff(positions_x)
    dy = np.diff(positions_y)
    msd = np.insert(np.cumsum(dx**2 + dy**2), 0, 0)
    return radius, radius_errors, positions_x, position_x_errors, positions_y, position_y_errors, msd

def save_plot_as_pdf(filename, plot_function, *args, **kwargs):
    """
    Helper function to save a single plot to a PDF.
    - filename : nom du fichier de sortie
    - plot_function : fonction de tracé à utiliser
    """
    with PdfPages(filename) as pdf:
        plot_function(*args, **kwargs)
        pdf.savefig()
    plt.show()

def plot_trajectory(x, y, x_err, y_err, image, title=None):
    """
    Trace la trajectoire du point avec barres d'erreur.
    - x, y : positions
    - x_err, y_err : erreurs
    - image : image de fond pour les dimensions
    - title : titre du graphique
    """
    height, width = image.shape
    plt.figure(figsize=(12, 8))
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o-', markersize=5, capsize=3, label='Position (± erreur)')
    plt.title(title, fontsize=16)
    plt.xlabel('Position X (pixels)', fontsize=14)
    plt.ylabel('Position Y (pixels)', fontsize=14)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

def plot_msd_with_regression(msd, times, title=None):
    """
    Trace le MSD avec la ligne de régression linéaire.
    - msd : Mean Squared Displacement
    - times : temps (frames)
    - title : titre du graphique
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(times, msd)
    regression_line = slope * times + intercept

    # Calculate intercept standard error
    n = len(times)
    mean_x = np.mean(times)
    ssx = np.sum((times - mean_x) ** 2)
    intercept_stderr = std_err * np.sqrt(1 / n + mean_x ** 2 / ssx)

    plt.figure(figsize=(12, 8))
    plt.plot(times, msd, 'o-', label='MSD', markersize=5)
    plt.plot(times, regression_line, 'r-', label=f'Régression linéaire: y={slope:.2f}x + {intercept:.2f}')
    plt.title(title, fontsize=16)
    plt.xlabel('Temps (frames)', fontsize=14)
    plt.ylabel('MSD (pixels²)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    return slope, intercept, r_value, std_err, intercept_stderr

def main():
    """
    Fonction principale pour traiter les fichiers TIFF et générer les graphiques.
    """
    save_dir = r"C:\Users\tomde\Desktop\Cours\Autonme 2024\Tech Ex\Mandat 3\Codes\Images"
    os.makedirs(save_dir, exist_ok=True)

    files = [
        (r"C:\Users\tomde\Downloads\Film_D_0.1_exp_0.5s_PixelSize_0.5um_8bit.tif", 8),
    ]
    for tiff_path, bit_depth in files:
        images, n_frames = load_tiff_sequence(tiff_path, bit_depth)
        radius, radius_errors, positions_x, position_x_errors, positions_y, position_y_errors, msd = compute_trajectory(images)
        times = np.arange(n_frames)

        radius_mean = np.mean(radius) * 0.5
        radius_error_mean = np.mean(radius_errors)
        print(f"Rayon moyen pour {bit_depth}-bit : {radius_mean:.4f} ± {radius_error_mean:.4f} μm")

        trajectory_pdf = os.path.join(save_dir, f"trajectory_{bit_depth}bit.pdf")
        save_plot_as_pdf(trajectory_pdf, plot_trajectory, positions_x, positions_y, position_x_errors, position_y_errors, images[0])

        msd_pdf = os.path.join(save_dir, f"msd_{bit_depth}bit.pdf")
        slope, intercept, r_value, std_err, intercept_stderr = plot_msd_with_regression(msd, times)
        save_plot_as_pdf(msd_pdf, plot_msd_with_regression, msd, times)

        print(f"Linear Regression Results for {bit_depth}-bit MSD:")
        print(f"  Slope: {slope:.4f} ± {std_err:.4f}")
        print(f"  Intercept: {intercept:.4f} ± {intercept_stderr:.4f}")
        print(f"  R-squared: {r_value**2:.4f}")

    print(f"Plots saved in {save_dir}")

if __name__ == "__main__":
    main()