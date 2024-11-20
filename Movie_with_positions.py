import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import lmfit
import os 

def load_tiff_sequence(tiff_path, bit_depth):
    """
    Charge une séquence d'images TIFF dans un tableau numpy 3D.
    - tiff_path : chemin vers le fichier TIFF
    - bit_depth : profondeur de bits (8 ou 16)
    """
    img = Image.open(tiff_path)
    n_frames = img.n_frames
    frames = []
    for i in range(n_frames):
        img.seek(i)
        frame = img.convert('L') if img.mode != 'L' else img
        frame_array = np.array(frame)
        normalize_factor = 65535.0 if bit_depth == 16 else 255.0
        frame_array = frame_array.astype(np.float64) / normalize_factor
        frames.append(frame_array)
    return np.array(frames)

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
    Effectue un ajustement Gaussien sur des données 2D et calcule les erreurs.
    - Z : image 2D à analyser
    Retourne les positions x0 et y0.
    """
    height, width = Z.shape
    y, x = np.indices(Z.shape)
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = Z.ravel()

    # Initial guess for Gaussian parameters
    initial_params = {
        'A': Z.max() - Z.min(),
        'x0': width / 2,
        'y0': height / 2,
        'sigma_x': width / 4,
        'sigma_y': height / 4,
        'B': Z.min(),
    }

    model = lmfit.Model(gaussian_2d, independent_vars=['x', 'y'])
    params = model.make_params(**initial_params)
    params['A'].min = 0
    params['sigma_x'].min = 1e-5
    params['sigma_y'].min = 1e-5
    params['x0'].set(min=0, max=width - 1)
    params['y0'].set(min=0, max=height - 1)

    result = model.fit(z_flat, params, x=x_flat, y=y_flat)
    return result.params['x0'].value, result.params['y0'].value

def compute_positions(frames):
    """
    Calcule les positions du point lumineux pour chaque image.
    - frames : séquence d'images 3D
    Retourne un tableau 2D des positions (x, y).
    """
    positions = []
    for frame in frames:
        x, y = lmfit_process(frame)
        positions.append((x, y))
    return np.array(positions)

def play_movie_with_positions(frames, positions, interval=100, save_path=None):
    """
    Joue et éventuellement sauvegarde une séquence d'images avec la trajectoire superposée.
    - frames : tableau numpy 3D de forme (n_frames, height, width)
    - positions : tableau numpy 2D de forme (n_frames, 2) contenant les positions x, y
    - interval : intervalle de temps entre les images en millisecondes
    - save_path : chemin pour sauvegarder l'animation. Si None, l'animation ne sera pas sauvegardée.
    """
    height, width = frames[0].shape

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(frames[0], origin='lower', cmap='gray', extent=(0, width, 0, height), alpha=0.8)
    
    trajectory, = ax.plot([], [], 'r-', lw=2, label="Trajectoire")
    current_position, = ax.plot([], [], 'bo', markersize=8, label="Position Actuelle")

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("Position X (pixels)", fontsize=14)
    ax.set_ylabel("Position Y (pixels)", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    
    def update(frame):
        img.set_data(frames[frame])
        trajectory.set_data(positions[:frame + 1, 0], positions[:frame + 1, 1])
        current_position.set_data([positions[frame, 0]], [positions[frame, 1]])
        return img, trajectory, current_position

    ani = FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=True, repeat=True
    )
    
    # Sauvegarder l'animation si un chemin est fourni
    if save_path:
        writer = FFMpegWriter(fps=1000 // interval)
        ani.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    
    plt.show()

def save_all_frames_as_pdfs(frames, positions, output_directory):
    """
    Sauvegarde toutes les images avec la trajectoire superposée en fichiers PDF individuels.
    - frames : tableau numpy 3D de forme (n_frames, height, width)
    - positions : tableau numpy 2D de forme (n_frames, 2) contenant les positions x, y
    - output_directory : répertoire pour sauvegarder les fichiers PDF
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Crée le répertoire s'il n'existe pas

    height, width = frames[0].shape

    for i, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(frame, origin='lower', cmap='gray', extent=(0, width, 0, height), alpha=0.8)
        ax.plot(positions[:i + 1, 0], positions[:i + 1, 1], 'r-', lw=2, label="Trajectoire")
        ax.plot(positions[i, 0], positions[i, 1], 'bo', markersize=8, label="Position Actuelle")

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel("Position X (pixels)", fontsize=14)
        ax.set_ylabel("Position Y (pixels)", fontsize=14)
        ax.legend(fontsize=12, loc="upper right")

        plt.tight_layout()

        # Sauvegarder l'image actuelle en PDF
        pdf_path = os.path.join(output_directory, f"frame-{i + 1}.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
            print(f"Frame {i + 1} saved to {pdf_path}")

        plt.close(fig)  # Fermer la figure pour économiser de la mémoire

def main():
    """
    Fonction principale pour charger les fichiers TIFF, calculer les positions, jouer le film et sauvegarder les images.
    """
    tiff_path = r"C:\Users\tomde\Downloads\Film_D_0.1_exp_0.5s_PixelSize_0.5um_8bit.tif"
    bit_depth = 8  # Changer à 16 si votre TIFF est en 16 bits
    frames = load_tiff_sequence(tiff_path, bit_depth)
    
    # Calculer les positions en utilisant lmfit
    positions = compute_positions(frames)
    
    # Jouer le film et le sauvegarder en fichier MP4
    save_path = r"C:\Users\tomde\Desktop\Cours\Autonme 2024\Tech Ex\Mandat 3\Codes\trajectory_animation.mp4"
    play_movie_with_positions(frames, positions, save_path=save_path)
    
    # Sauvegarder toutes les images avec la trajectoire en fichiers PDF
    output_directory = r"C:\Users\tomde\Desktop\Cours\Autonme 2024\Tech Ex\Mandat 3\Codes\ImagesGIF"
    save_all_frames_as_pdfs(frames, positions, output_directory)

if __name__ == "__main__":
    main()