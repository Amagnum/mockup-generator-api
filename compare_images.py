import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import argparse
from pathlib import Path
import itertools
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec

def load_image(image_path):
    """Load an image and convert to grayscale numpy array."""
    img = Image.open(image_path)
    return img

def get_comparison_array(img):
    """Convert image to grayscale array for comparison metrics."""
    if img.mode != 'L':
        gray_img = img.convert('L')
    else:
        gray_img = img
    return np.array(gray_img)

def compare_images(img1, img2):
    """Compare two images using MSE and SSIM metrics."""
    # Get grayscale arrays for comparison
    gray1 = get_comparison_array(img1)
    gray2 = get_comparison_array(img2)
    
    # Resize images if they have different dimensions
    if gray1.shape != gray2.shape:
        gray2 = np.array(Image.fromarray(gray2).resize(
            (gray1.shape[1], gray1.shape[0]), Image.LANCZOS))
    
    # Calculate metrics
    mse_value = mse(gray1, gray2)
    ssim_value = ssim(gray1, gray2)
    
    return {
        'mse': mse_value,
        'ssim': ssim_value
    }

def display_images_with_sync_zoom(images, image_paths):
    """Display multiple images with synchronized zooming and panning."""
    n_images = len(images)
    
    # Determine grid layout based on number of images
    if n_images <= 2:
        rows, cols = 1, n_images
    elif n_images <= 4:
        rows, cols = 2, 2
    else:  # 5 images
        rows, cols = 2, 3
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(cols*5, rows*5))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    # Create subplots
    axes = []
    for i in range(n_images):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(np.array(images[i]))
        ax.set_title(f"{image_paths[i].name}")
        ax.axis('on')  # Keep axis on for better zoom control
        axes.append(ax)
    
    # Create a class to handle synchronized zooming
    class SyncZoom:
        def __init__(self, axes):
            self.axes = axes
            self.currently_zooming = False
            self.press = None
            self.background = None
            
            # Connect events
            for ax in self.axes:
                ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
                ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
                ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
                ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        def on_press(self, event):
            if event.inaxes is None:
                return
            self.press = event.xdata, event.ydata
            self.currently_zooming = True
        
        def on_release(self, event):
            self.press = None
            self.currently_zooming = False
        
        def on_motion(self, event):
            if not self.currently_zooming or self.press is None or event.inaxes is None:
                return
            
            # Calculate the movement
            dx = event.xdata - self.press[0]
            dy = event.ydata - self.press[1]
            
            # Update all axes
            for ax in self.axes:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
                ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            
            # Update the press point
            self.press = event.xdata, event.ydata
            
            # Redraw the figure
            fig.canvas.draw_idle()
        
        def on_scroll(self, event):
            if event.inaxes is None:
                return
            
            # Get the current axis limits
            ax = event.inaxes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Get the current mouse position
            x = event.xdata
            y = event.ydata
            
            # Zoom factor
            scale_factor = 1.1 if event.button == 'up' else 1/1.1
            
            # Set new limits
            new_width = (xlim[1] - xlim[0]) / scale_factor
            new_height = (ylim[1] - ylim[0]) / scale_factor
            
            # Set new limits for all axes
            for ax in self.axes:
                # Calculate new limits centered on mouse position
                xlim_new = [x - new_width * (x - xlim[0]) / (xlim[1] - xlim[0]),
                           x + new_width * (xlim[1] - x) / (xlim[1] - xlim[0])]
                ylim_new = [y - new_height * (y - ylim[0]) / (ylim[1] - ylim[0]),
                           y + new_height * (ylim[1] - y) / (ylim[1] - ylim[0])]
                
                ax.set_xlim(xlim_new)
                ax.set_ylim(ylim_new)
            
            # Redraw the figure
            fig.canvas.draw_idle()
    
    # Initialize the synchronizer
    sync_zoom = SyncZoom(axes)
    
    # Add reset zoom button
    plt.subplots_adjust(bottom=0.15)
    reset_ax = plt.axes([0.45, 0.05, 0.1, 0.05])
    reset_button = Button(reset_ax, 'Reset Zoom')
    
    def reset_zoom(event):
        for ax in axes:
            ax.autoscale(True)
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()
    
    reset_button.on_clicked(reset_zoom)
    
    # Add instructions
    fig.text(0.5, 0.01, 
             "Instructions: Scroll to zoom, click and drag to pan, press Reset Zoom button to reset view",
             ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

def display_image_pair(img1, img2, path1, path2, metrics):
    """Display two images side by side with comparison metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display images
    ax1.imshow(np.array(img1))
    ax1.set_title(f"{path1.name}")
    ax1.axis('off')
    
    ax2.imshow(np.array(img2))
    ax2.set_title(f"{path2.name}")
    ax2.axis('off')
    
    # Add metrics as suptitle
    plt.suptitle(f"MSE: {metrics['mse']:.2f} | SSIM: {metrics['ssim']:.4f}", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare all images in a folder')
    parser.add_argument('folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--threshold', type=float, default=0.9, 
                        help='SSIM threshold for similarity (0-1, default: 0.9)')
    parser.add_argument('--display', action='store_true', 
                        help='Display image pairs side by side')
    parser.add_argument('--display-similar', action='store_true',
                        help='Only display image pairs that are similar (above threshold)')
    parser.add_argument('--sync-view', action='store_true',
                        help='Display all images with synchronized zooming and panning')
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [f for f in folder_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if len(image_files) < 2:
        print(f"Found only {len(image_files)} images. Need at least 2 for comparison.")
        return
    
    # Limit to 5 images maximum
    if len(image_files) > 5:
        print(f"Found {len(image_files)} images. Using only the first 5 for synchronized view.")
        image_files = image_files[:5]
    else:
        print(f"Found {len(image_files)} images.")
    
    # Load all images
    images = {}
    for img_path in image_files:
        try:
            images[img_path] = load_image(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # If sync-view is enabled, display all images with synchronized zooming
    if args.sync_view:
        display_images_with_sync_zoom(
            [images[path] for path in image_files if path in images],
            [path for path in image_files if path in images]
        )
        return
    
    # Compare all pairs
    similar_pairs = []
    for (path1, img1), (path2, img2) in itertools.combinations(images.items(), 2):
        metrics = compare_images(img1, img2)
        
        # Print comparison results
        print(f"\nComparing {path1.name} and {path2.name}:")
        print(f"  MSE: {metrics['mse']:.2f} (lower is more similar)")
        print(f"  SSIM: {metrics['ssim']:.4f} (higher is more similar)")
        
        # Track similar images
        is_similar = metrics['ssim'] > args.threshold
        if is_similar:
            similar_pairs.append((path1, path2, metrics['ssim'], metrics))
        
        # Display images if requested
        if args.display and (not args.display_similar or is_similar):
            display_image_pair(img1, img2, path1, path2, metrics)
    
    # Report similar images
    if similar_pairs:
        print(f"\nPotentially similar image pairs (SSIM > {args.threshold}):")
        for path1, path2, similarity, _ in sorted(similar_pairs, key=lambda x: x[2], reverse=True):
            print(f"  {path1.name} and {path2.name}: SSIM = {similarity:.4f}")
    else:
        print(f"\nNo image pairs with similarity above {args.threshold} threshold found.")

if __name__ == "__main__":
    main() 