import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import tifffile
from scipy.optimize import curve_fit
from scipy import stats

def improve_background_and_enhance_psf(frame, intensity_ratio=7.17, verbose=False):
    """
    Enhance PSFs and darken background with less aggressive threshold logic.
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = frame[:,:,0]
    else:
        gray = frame
    
    # Store original intensities
    original_intensities = gray.copy()
    
    # Background correction
    background = cv2.GaussianBlur(gray, (27, 27), 0)
    corrected = cv2.subtract(gray, background)
    
    # Enhance PSFs
    enhanced = corrected
    
    # Find maximum intensity
    max_intensity = np.max(enhanced)
    
    # Calculate background threshold
    background_threshold = max_intensity / (1 + intensity_ratio)
    
    # Create refined binary mask
    refined_mask = (enhanced > background_threshold).astype(np.uint8) * 255
    
    # Apply mask to get final result
    result = cv2.bitwise_and(enhanced, enhanced, mask=refined_mask)
    
    # Normalize the result for display
    normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if verbose:
        print(f"Frame statistics:")
        print(f"Max intensity: {max_intensity}")
        print(f"Threshold (max/{1+intensity_ratio}): {background_threshold}")
        return normalized, background_threshold, max_intensity
    return normalized

def check_displacement(positions, min_displacement=0.02):
    """Check if a PSF shows significant displacement."""
    positions = np.array(positions)
    if len(positions) < 2:
        return False
    
    # Calculate total displacement (distance between start and end)
    total_displacement = np.sqrt(np.sum((positions[-1] - positions[0])**2))
    
    # Calculate average frame-to-frame displacement
    displacements = np.sqrt(np.sum((positions[1:] - positions[:-1])**2, axis=1))
    avg_displacement = np.mean(displacements)
    
    # Check if trajectory is long enough
    min_frames = 35
    
    return (len(positions) >= min_frames and 
            total_displacement >= min_displacement and 
            avg_displacement >= 0.02)

def track_psfs(frames, magnification_factor=10):
    tracks = []
    max_distance = 8
    min_intensity = 30
    
    for frame_idx, frame in enumerate(frames):
        enhanced_frame = improve_background_and_enhance_psf(frame)
        binary = (enhanced_frame > min_intensity).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_positions = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                current_positions.append((cx, cy))
        
        if frame_idx == 0:
            for pos in current_positions:
                tracks.append({
                    'positions': [pos],
                    'frames': [0],
                    'active': True
                })
        else:
            if current_positions and tracks:
                active_tracks = [t for t in tracks if t['active']]
                if active_tracks:
                    last_positions = [t['positions'][-1] for t in active_tracks]
                    scaled_max_distance = max_distance
                    distances = cdist(last_positions, current_positions)
                    
                    while distances.size > 0:
                        min_dist = np.min(distances)
                        if min_dist > scaled_max_distance:
                            break
                        
                        i, j = np.unravel_index(np.argmin(distances), distances.shape)
                        active_tracks[i]['positions'].append(current_positions[j])
                        active_tracks[i]['frames'].append(frame_idx)
                        
                        distances = np.delete(distances, i, axis=0)
                        distances = np.delete(distances, j, axis=1)
                        current_positions.pop(j)
                        active_tracks.pop(i)
            
            for pos in current_positions:
                tracks.append({
                    'positions': [pos],
                    'frames': [frame_idx],
                    'active': True
                })
    
    return tracks

def calculate_msd(positions, dt=0.033):
    """Calculate Mean Square Displacement."""
    positions = np.array(positions)
    n_points = len(positions)
    max_tau = n_points - 1
    
    msds = []
    times = []
    
    for tau in range(1, max_tau):
        diffs = positions[tau:] - positions[:-tau]
        squares = np.sum(diffs**2, axis=1)
        msd = np.mean(squares)
        
        msds.append(msd)
        times.append(tau * dt)
    
    return np.array(times), np.array(msds)

def trajectory_drift_correction(tracks, dt=0.033):
    """
    Implements drift correction by subtracting mean velocity from each trajectory
    """
    corrected_tracks = []
    
    for track in tracks:
        positions = np.array(track['positions'])
        frames = np.array(track['frames'])
        
        if len(positions) > 1:  # Only correct if we have more than one point
            # Calculate velocities between consecutive frames
            velocities = positions[1:] - positions[:-1]
            
            # Calculate mean velocity for this trajectory
            mean_velocity = np.mean(velocities, axis=0)
            
            # Create corrected positions by subtracting accumulated drift
            time_points = np.arange(len(positions))[:, np.newaxis]
            drift = mean_velocity * time_points
            corrected_positions = positions - drift
            
            # Create corrected track
            corrected_track = track.copy()
            corrected_track['positions'] = list(corrected_positions)
        else:
            corrected_track = track.copy()
            
        corrected_tracks.append(corrected_track)
    
    return corrected_tracks

def plot_msds(tracks, dt=0.033):
    # Apply trajectory-based drift correction
    corrected_tracks = trajectory_drift_correction(tracks)
    
    # Create single plot for MSDs with quadratic fit
    plt.figure(figsize=(10, 8))
    
    mobile_tracks = []
    for i, track in enumerate(corrected_tracks):
        positions = np.array(track['positions'])
        
        if check_displacement(positions):
            mobile_tracks.append(track)
            times, msd = calculate_msd(positions, dt)
            
            # Generate color for this track
            color = plt.cm.rainbow(i / len(tracks))
            
            # Plot individual MSD
            plt.plot(times, msd, 'o-', alpha=0.5, markersize=2, color=color)
            
            # Quadratic fit
            coeffs = np.polyfit(times, msd, 2)
            poly = np.poly1d(coeffs)
            fit_times = np.linspace(times[0], times[-1], 100)
            plt.plot(fit_times, poly(fit_times), '--', alpha=0.5, color=color,
                     label=f'Track {i}: D = {coeffs[2]:.4f}t² + {coeffs[1]:.4f}t + {coeffs[0]:.4f}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (pixels²)')
    plt.title(f'Individual MSDs with Trajectory-based Drift Correction (n={len(mobile_tracks)} mobile PSFs)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return mobile_tracks

def plot_drift_correction_validation(tracks, dt=0.033):
    """
    Plot validation of drift correction showing before/after trajectories
    Parameters:
        tracks: list of track dictionaries
        dt: time step between frames in seconds
    """
    corrected_tracks = trajectory_drift_correction(tracks)
    
    # First figure: Trajectories
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    
    # Plot trajectories before and after correction
    for track, corrected_track in zip(tracks[:5], corrected_tracks[:5]):
        # Multiply positions by 10 to restore original pixel positions
        pos = np.array(track['positions']) * 10
        corrected_pos = np.array(corrected_track['positions']) * 10
        
        ax1.plot(pos[:, 0], pos[:, 1], 'r-', alpha=0.5, label='Original')
        ax1.plot(corrected_pos[:, 0], corrected_pos[:, 1], 'b-', alpha=0.5, label='Corrected')
    
    ax1.set_title('Sample Trajectories\nBefore (red) and After (blue) Correction')
    ax1.set_xlabel('Position (pixels)')
    ax1.set_ylabel('Position (pixels)')
    
    # Set specific axis limits
    ax1.set_xlim(0, 1280)
    ax1.set_ylim(1024, 0)  # Reversed y-axis with specific limits
    
    # Add grid to first plot
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    
    # Second figure: MSD curves
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    
    # Plot MSD curves before and after correction
    for track in tracks[:5]:
        times, msd = calculate_msd(np.array(track['positions']), dt)
        ax2.plot(times, msd, 'r-', alpha=0.5)
        
    for track in corrected_tracks[:5]:
        times, msd = calculate_msd(np.array(track['positions']), dt)
        ax2.plot(times, msd, 'b-', alpha=0.5)
    
    ax2.set_title('Sample MSDs\nBefore (red) and After (blue) Correction')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('MSD (pixels²)')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_all_trajectories(tracks):
    analyzed_tracks = []
    excluded_tracks = [2, 3, 5]
    
    for i, track in enumerate(tracks):
        if i in excluded_tracks or len(track['positions']) < 2:
            continue
            
        positions = np.array(track['positions'])
        frames = np.array(track['frames'])
        velocities = positions[1:] - positions[:-1]
        mean_velocity = np.mean(velocities, axis=0)
        time_points = np.arange(len(positions))[:, np.newaxis]
        drift = mean_velocity * time_points
        corrected_positions = positions - drift
        
        analyzed_tracks.append({
            'initial_pos': positions[0],
            'positions': positions,
            'corrected_positions': corrected_positions,
            'corrected_x': corrected_positions[:,0],
            'corrected_y': corrected_positions[:,1],
            'frames': frames,
            'drift': drift
        })
    
    return analyzed_tracks
def plot_trajectories_vs_time(analyzed_tracks, dt=0.033):
    # Plot X trajectory
    plt.figure(figsize=(8, 6))
    track = analyzed_tracks[2]  # Get track 1
    times = np.arange(len(track['frames'])) * dt
    
    corr_x = track['corrected_positions'][:,0]
    plt.plot(times, corr_x, '--', label='Corrected')
    
    plt.title('X Position vs Time (Track 2)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pixels)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot Y trajectory
    plt.figure(figsize=(8, 6))
    corr_y = track['corrected_positions'][:,1]
    plt.plot(times, corr_y, '--', label='Corrected')
    
    plt.title('Y Position vs Time (Track 2)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pixels)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()





def plot_msd_from_corrected(analyzed_tracks, dt=0.033):
    plt.figure(figsize=(10, 6))
    all_msds = []
    D_values = []
    D_errors = []
    min_length = float('inf')
    
    # First pass to get minimum length and collect MSDs
    for i, track in enumerate(analyzed_tracks):
        if len(track['frames']) >= 50 and i != 13:
            positions = track['corrected_positions']
            n_points = len(positions)
            max_tau = n_points - 1
            msds = []
            times = []
            
            for tau in range(1, max_tau):
                diffs = positions[tau:] - positions[:-tau]
                squares = np.sum(diffs**2, axis=1)
                msd = np.mean(squares)
                msds.append(msd)
                times.append(tau * dt)
            
            min_length = min(min_length, len(msds))
            all_msds.append(msds[:min_length])
            
            times = np.array(times[:min_length])
            msds = np.array(msds[:min_length])
            
            fit_range = times <= 5* dt
            fit_times = times[fit_range]
            fit_msds = msds[fit_range]
            
            # Calculate linear fit with scipy.stats for error estimation
            slope, intercept, r_value, p_value, slope_std_error = stats.linregress(fit_times, fit_msds)
            D = slope/4
            D_error = slope_std_error/4
            
            D_values.append(D)
            D_errors.append(D_error)
            
            fit_line = slope * fit_times + intercept
            r2 = r_value**2
            
            plt.plot(times, msds, '-', alpha=0.3, 
                    label=f'Track {i} (D={D:.2f}±{D_error:.2f} pixels²/s, R²={r2:.3f})')
            plt.plot(fit_times, fit_line, '--', alpha=0.3)
    
    # Calculate mean and standard error of D values
    mean_D = np.mean(D_values)
    std_error_D = np.std(D_values) / np.sqrt(len(D_values))
    
    # Calculate mean MSD
    all_msds = np.array(all_msds)
    mean_msd = np.mean(all_msds, axis=0)
    times = np.array([t * dt for t in range(1, len(mean_msd) + 1)])
    
    # Fit mean MSD
    fit_range_mean = times <= 5 * dt
    fit_times_mean = times[fit_range_mean]
    fit_msds_mean = mean_msd[fit_range_mean]
    
    slope_mean, intercept_mean, r_value_mean, p_value_mean, slope_std_error_mean = stats.linregress(fit_times_mean, fit_msds_mean)
    fit_line_mean = slope_mean * fit_times_mean + intercept_mean
    D_mean = slope_mean/4
    D_mean_error = slope_std_error_mean/4
    r2_mean = r_value_mean**2
    
    # Plot mean MSD
    plt.plot(times, mean_msd, 'k-', linewidth=2, 
            label=f'Mean MSD (D={D_mean:.2f}±{D_mean_error:.2f} pixels²/s, R²={r2_mean:.3f})')
    plt.plot(fit_times_mean, fit_line_mean, 'r--', linewidth=2)
    
    # Print results
    print("\nDiffusion Coefficient Analysis:")
    print("-" * 40)
    print(f"Individual track D values (pixels²/s) and R² values:")
    for i, (D, D_err) in enumerate(zip(D_values, D_errors)):
        print(f"Track {i}: D = {D:.3f} ± {D_err:.3f}, R² = {r2:.3f}")
    print("-" * 40)
    print(f"Mean D from individual tracks = {mean_D:.3f} ± {std_error_D:.3f} pixels²/s")
    print(f"D from mean MSD = {D_mean:.3f} ± {D_mean_error:.3f} pixels²/s, R² = {r2_mean:.3f}")
    
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (pixels²)')
    plt.title('MSD from Corrected Positions')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return mean_D, std_error_D, D_mean, D_mean_error, r2_mean

def main():
    # Load frames
    frames = tifffile.imread("1microns.tif")
    # Track PSFs
    print("Tracking PSFs...")
    tracks = track_psfs(frames)
    print(f"Found {len(tracks)} total tracks")
    analyzed=analyze_all_trajectories(tracks)
    print(f"Number of analyzed tracks: {len(analyzed)}")
    mean_D, std_error_D, D_mean, D_mean_error= plot_msd_from_corrected(analyzed)
    plot_trajectories_vs_time(analyzed)
    for i, track in enumerate(analyzed):
        print(f"Track {i}: {len(track['frames'])} frames")
        print(f"\nTrack {i}")
        print(f"Initial position: ({track['initial_pos'][0]:.1f}, {track['initial_pos'][1]:.1f})")
        print(f"Positions range X: {np.min(track['positions'][:,0]):.1f} to {np.max(track['positions'][:,0]):.1f}")
        print(f"Positions range Y: {np.min(track['positions'][:,1]):.1f} to {np.max(track['positions'][:,1]):.1f}")
        print(f"Corrected positions range X: {np.min(track['corrected_positions'][:,0]):.1f} to {np.max(track['corrected_positions'][:,0]):.1f}")
        print(f"Corrected positions range Y: {np.min(track['corrected_positions'][:,1]):.1f} to {np.max(track['corrected_positions'][:,1]):.1f}")
        print(f"Max drift: ({np.max(track['drift'][:,0]):.1f}, {np.max(track['drift'][:,1]):.1f})")

if __name__ == "__main__":
    main()
