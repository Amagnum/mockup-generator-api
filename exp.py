import numpy as np
import cv2
import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image

def load_depth_map(path, normalize=True):
    """
    Load a 16-bit depth map and optionally normalize it.
    
    Args:
        path: Path to the depth map image
        normalize: Whether to normalize the depth values to [0, 1]
        
    Returns:
        Depth map as numpy array
    """
    # Load the 16-bit depth map
    depth_map = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    
    if depth_map is None:
        raise ValueError(f"Could not load depth map from {path}")
    
    # Normalize if requested
    if normalize:
        # Convert to float and normalize to [0, 1]
        depth_map = depth_map.astype(np.float32) / 65535.0
    
    return depth_map

def load_design(path, target_size=None):
    """
    Load a design image and resize it if needed.
    
    Args:
        path: Path to the design image
        target_size: Optional (width, height) to resize the design
        
    Returns:
        Design image as numpy array with RGBA channels
    """
    # Load the design with alpha channel if available
    design = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if design is None:
        raise ValueError(f"Could not load design from {path}")
    
    # Convert BGR to RGB
    if design.shape[2] == 3:
        design = cv2.cvtColor(design, cv2.COLOR_BGR2RGB)
        # Add alpha channel
        alpha = np.ones((design.shape[0], design.shape[1], 1), dtype=design.dtype) * 255
        design = np.concatenate([design, alpha], axis=2)
    elif design.shape[2] == 4:
        design = cv2.cvtColor(design, cv2.COLOR_BGRA2RGBA)
    
    # Resize if target size is provided
    if target_size is not None:
        design = cv2.resize(design, target_size, interpolation=cv2.INTER_AREA)
    
    return design

def detect_tshirt_region(depth_map, threshold=0.5):
    """
    Detect the t-shirt region in the depth map.
    
    Args:
        depth_map: Normalized depth map
        threshold: Threshold for segmentation
        
    Returns:
        Mask of the t-shirt region
    """
    # Simple thresholding to find the t-shirt region
    # Adjust the threshold based on your specific depth map
    mask = (depth_map > threshold).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def create_mesh_from_depth(depth_map, mask=None, downsample=1):
    """
    Create a 3D mesh from the depth map.
    
    Args:
        depth_map: Depth map as numpy array
        mask: Optional mask to specify the region of interest
        downsample: Factor to downsample the mesh (for efficiency)
        
    Returns:
        vertices: List of (x, y, z) coordinates
        faces: List of triangle indices
        tex_coords: List of (u, v) texture coordinates
    """
    height, width = depth_map.shape
    vertices = []
    faces = []
    tex_coords = []
    
    # Create vertices and texture coordinates
    for y in range(0, height, downsample):
        for x in range(0, width, downsample):
            if mask is None or mask[y, x] > 0:
                # Depth value (z-coordinate)
                z = depth_map[y, x]
                
                # Add vertex
                vertices.append((x, y, z))
                
                # Add texture coordinate (normalized to [0, 1])
                tex_coords.append((x / width, y / height))
    
    # Create faces (triangles)
    vertices_map = {}
    vertex_count = 0
    
    for y in range(0, height - downsample, downsample):
        for x in range(0, width - downsample, downsample):
            # Check if all four corners are valid
            valid = True
            if mask is not None:
                for dy in range(2):
                    for dx in range(2):
                        if mask[y + dy * downsample, x + dx * downsample] == 0:
                            valid = False
                            break
            
            if valid:
                # Get indices of the four corners
                indices = []
                for dy in range(2):
                    for dx in range(2):
                        pos = (x + dx * downsample, y + dy * downsample)
                        if pos not in vertices_map:
                            vertices_map[pos] = vertex_count
                            vertex_count += 1
                        indices.append(vertices_map[pos])
                
                # Create two triangles
                faces.append((indices[0], indices[1], indices[2]))
                faces.append((indices[1], indices[3], indices[2]))
    
    return vertices, faces, tex_coords

def apply_design_to_mesh(depth_map, design, output_path, mask=None, downsample=2):
    """
    Apply a design to a 3D mesh created from a depth map.
    
    Args:
        depth_map: Depth map as numpy array
        design: Design image as numpy array
        output_path: Path to save the result
        mask: Optional mask to specify the region of interest
        downsample: Factor to downsample the mesh (for efficiency)
    """
    # Create mesh from depth map
    vertices, faces, tex_coords = create_mesh_from_depth(depth_map, mask, downsample)
    
    # Initialize OpenGL
    import sys
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(depth_map.shape[1], depth_map.shape[0])
    window = glutCreateWindow(b"T-Shirt Design Mapping")  # Use bytes instead of string
    
    # Set up OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Create texture from design
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Make sure design is in the right format for OpenGL
    if design.dtype != np.uint8:
        design = (design * 255).astype(np.uint8)
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, design.shape[1], design.shape[0], 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, design)
    
    # Set up projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, depth_map.shape[1] / depth_map.shape[0], 0.1, 100.0)
    
    # Set up camera
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(depth_map.shape[1] / 2, depth_map.shape[0] / 2, -max(depth_map.shape),
              depth_map.shape[1] / 2, depth_map.shape[0] / 2, 0,
              0, -1, 0)
    
    # Clear the screen
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw the mesh with the design texture
    glBegin(GL_TRIANGLES)
    for face in faces:
        for i, vertex_idx in enumerate(face):
            glTexCoord2f(tex_coords[vertex_idx][0], tex_coords[vertex_idx][1])
            glVertex3f(vertices[vertex_idx][0], vertices[vertex_idx][1], vertices[vertex_idx][2])
    glEnd()
    
    # Read the rendered image
    width, height = depth_map.shape[1], depth_map.shape[0]
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    
    # Flip the image vertically (OpenGL convention)
    image = np.flipud(image)
    
    # Save the result
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    
    # Clean up
    glutDestroyWindow(window)

def load_mask(path):
    """
    Load a mask image.
    
    Args:
        path: Path to the mask image
        
    Returns:
        Mask as numpy array
    """
    # Load the mask image
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"Could not load mask from {path}")
    
    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask

def apply_design_to_mesh_alternative(depth_map, design, output_path, mask=None, downsample=2, 
                                     debug=True, depth_intensity=1.0, normal_intensity=0.05):
    """
    Alternative implementation that doesn't rely on OpenGL.
    Maps a design onto a t-shirt using the depth map.
    
    Args:
        depth_map: Depth map as numpy array
        design: Design image as numpy array
        output_path: Path to save the result
        mask: Optional mask to specify the region of interest
        downsample: Factor to downsample the mesh (for efficiency)
        debug: Whether to save intermediate results for debugging
        depth_intensity: Controls how much the depth affects the mapping (0.0-2.0)
        normal_intensity: Controls how much the surface normals affect the mapping (0.0-0.2)
    """
    print(f"Starting alternative mesh mapping method with depth_intensity={depth_intensity}, normal_intensity={normal_intensity}...")
    height, width = depth_map.shape
    result = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Normalize depth for better visualization
    depth_normalized = depth_map.copy()
    if mask is not None:
        print(f"Using mask with {np.sum(mask > 0)} non-zero pixels")
        # Only consider depths within the mask
        valid_depths = depth_map[mask > 0]
        if len(valid_depths) > 0:
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            print(f"Depth range in masked region: {min_depth:.4f} to {max_depth:.4f}")
            if max_depth > min_depth:
                depth_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    
    # Apply depth intensity
    if depth_intensity != 1.0:
        # Apply non-linear transformation to depth values
        depth_normalized = np.power(depth_normalized, depth_intensity)
        print(f"Applied depth intensity {depth_intensity}")
    
    if debug:
        # Save normalized depth map
        cv2.imwrite("debug_normalized_depth.png", (depth_normalized * 255).astype(np.uint8))
        print("Saved normalized depth map to debug_normalized_depth.png")
    
    print("Calculating surface normals...")
    # Calculate surface normals (approximation)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            if mask is None or mask[y, x] > 0:
                # Calculate partial derivatives
                dx = (depth_normalized[y, x+1] - depth_normalized[y, x-1]) / 2.0
                dy = (depth_normalized[y+1, x] - depth_normalized[y-1, x]) / 2.0
                
                # Normal vector (-dx, -dy, 1) normalized
                normal = np.array([-dx, -dy, 1.0])
                norm = np.sqrt(np.sum(normal**2))
                if norm > 0:
                    normal = normal / norm
                
                normals[y, x] = normal
    
    if debug:
        # Visualize normals as RGB
        normal_vis = ((normals + 1) / 2 * 255).astype(np.uint8)
        cv2.imwrite("debug_normals.png", cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))
        print("Saved surface normals visualization to debug_normals.png")
    
    print("Mapping design to surface...")
    # Map design to the surface
    design_height, design_width = design.shape[:2]
    print(f"Design dimensions: {design_width}x{design_height}")
    
    # Create a distortion map for visualization
    if debug:
        distortion_map = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Counter for progress reporting
    total_pixels = (height // downsample) * (width // downsample)
    processed_pixels = 0
    last_percent = 0
    
    for y in range(0, height, downsample):
        for x in range(0, width, downsample):
            if mask is None or mask[y, x] > 0:
                # Get texture coordinates
                u = x / width
                v = y / height
                
                # Apply some distortion based on depth and normals
                if y > 0 and y < height-1 and x > 0 and x < width-1:
                    normal = normals[y, x]
                    depth_factor = depth_normalized[y, x]
                    
                    # Adjust texture coordinates based on surface orientation and depth
                    u_offset = normal[0] * normal_intensity * depth_factor
                    v_offset = normal[1] * normal_intensity * depth_factor
                    
                    u = u + u_offset
                    v = v + v_offset
                    
                    if debug:
                        # Visualize distortion (red = u offset, green = v offset)
                        distortion_map[y, x, 0] = int((u_offset + normal_intensity) * 2550 / normal_intensity)
                        distortion_map[y, x, 1] = int((v_offset + normal_intensity) * 2550 / normal_intensity)
                        distortion_map[y, x, 2] = int(depth_factor * 255)
                
                # Clamp texture coordinates
                u = max(0, min(0.999, u))
                v = max(0, min(0.999, v))
                
                # Get pixel from design
                design_x = int(u * design_width)
                design_y = int(v * design_height)
                
                if design_x < design_width and design_y < design_height:
                    design_pixel = design[design_y, design_x]
                    
                    # Apply design with alpha blending
                    if len(design_pixel) == 4:
                        alpha = design_pixel[3] / 255.0
                        result[y, x, :3] = design_pixel[:3]
                        result[y, x, 3] = design_pixel[3]
            
            # Update progress
            processed_pixels += 1
            percent_done = (processed_pixels * 100) // total_pixels
            if percent_done > last_percent and percent_done % 10 == 0:
                print(f"Mapping progress: {percent_done}%")
                last_percent = percent_done
    
    if debug:
        cv2.imwrite("debug_distortion_map.png", distortion_map)
        print("Saved distortion map to debug_distortion_map.png")
        
        # Save raw result before filtering
        cv2.imwrite("debug_raw_result.png", result)
        print("Saved raw result to debug_raw_result.png")
    
    print("Applying median filter to smooth the result...")
    # Apply median filter to smooth the result
    result = cv2.medianBlur(result, 3)
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Final result saved to {output_path}")
    
    return result

def map_design_to_tshirt(depth_map_path, design_path, output_path, mask_path=None,
                         design_region=None, downsample=2, debug=True, 
                         depth_intensity=1.0, normal_intensity=0.05):
    """
    Main function to map a design onto a t-shirt using a depth map.
    
    Args:
        depth_map_path: Path to the 16-bit depth map
        design_path: Path to the design image
        output_path: Path to save the result
        mask_path: Path to the mask image (instead of auto-detection)
        design_region: Optional (x, y, width, height) to specify where to place the design
        downsample: Factor to downsample the mesh (for efficiency)
        debug: Whether to save intermediate results for debugging
        depth_intensity: Controls how much the depth affects the mapping (0.0-2.0)
        normal_intensity: Controls how much the surface normals affect the mapping (0.0-0.2)
    """
    print(f"Loading depth map from {depth_map_path}...")
    # Load depth map
    depth_map = load_depth_map(depth_map_path)
    print(f"Depth map loaded: {depth_map.shape}, range: {depth_map.min():.4f} to {depth_map.max():.4f}")
    
    # Load mask if provided, otherwise detect t-shirt region
    if mask_path:
        print(f"Loading mask from {mask_path}...")
        mask = load_mask(mask_path)
        print(f"Mask loaded: {mask.shape}, {np.sum(mask > 0)} non-zero pixels")
    else:
        print("Detecting t-shirt region...")
        mask = detect_tshirt_region(depth_map)
        print(f"T-shirt region detected: {np.sum(mask > 0)} pixels")
    
    if debug:
        cv2.imwrite("debug_mask.png", mask)
        print("Saved mask to debug_mask.png")
    
    print(f"Loading design from {design_path}...")
    # Load design
    design = load_design(design_path)
    print(f"Design loaded: {design.shape}")
    
    # If design region is specified, resize and position the design
    if design_region is not None:
        print(f"Positioning design in region: {design_region}")
        x, y, w, h = design_region
        design_resized = cv2.resize(design, (w, h), interpolation=cv2.INTER_AREA)
        print(f"Design resized to: {design_resized.shape}")
        
        # Create a blank image with the same size as the depth map
        full_design = np.zeros((depth_map.shape[0], depth_map.shape[1], 4), dtype=np.uint8)
        
        # Place the design at the specified position
        full_design[y:y+h, x:x+w] = design_resized
        design = full_design
        
        if debug:
            cv2.imwrite("debug_positioned_design.png", cv2.cvtColor(design, cv2.COLOR_RGBA2BGRA))
            print("Saved positioned design to debug_positioned_design.png")
    
    # Try the alternative implementation that doesn't use OpenGL
    try:
        result = apply_design_to_mesh_alternative(
            depth_map, design, output_path, mask, downsample, debug, 
            depth_intensity, normal_intensity
        )
        print(f"Design mapped to t-shirt using alternative method and saved to {output_path}")
    except Exception as e:
        print(f"Alternative method failed: {e}")
        try:
            # Fall back to the original method
            apply_design_to_mesh(depth_map, design, output_path, mask, downsample)
            print(f"Design mapped to t-shirt using original method and saved to {output_path}")
        except Exception as e:
            print(f"Both methods failed: {e}")
            raise

def visualize_result(original_path, depth_map_path, result_path):
    """
    Visualize the original image, depth map, and result side by side.
    
    Args:
        original_path: Path to the original image
        depth_map_path: Path to the depth map
        result_path: Path to the result image
    """
    # Load images
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    depth_map = load_depth_map(depth_map_path)
    
    result = cv2.imread(result_path, cv2.IMREAD_UNCHANGED)
    result = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(depth_map, cmap="gray")
    axes[1].set_title("Depth Map")
    axes[1].axis("off")
    
    axes[2].imshow(result)
    axes[2].set_title("Result")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    depth_map_path = "depth_map.png"
    design_path = "design.png"
    output_path = "output.png"
    mask_path = "mask.png"  # Add this line to use a mask image
    
    # Optional: specify the region where to place the design
    # Format: (x, y, width, height)
    design_region = (600, 600, 600, 600)
    
    # Intensity controls
    depth_intensity = -5  # Range: 0.0 (flat) to 2.0 (exaggerated)
    normal_intensity = 0.02  # Range: 0.0 (no distortion) to 0.2 (high distortion)
    
    # Use mask_path parameter with intensity controls
    map_design_to_tshirt(
        depth_map_path, design_path, output_path, mask_path, design_region, 
        downsample=2, debug=True, depth_intensity=depth_intensity, normal_intensity=normal_intensity
    )
    
    # Visualize the result
    original_path = "design2.jpg"
    visualize_result(original_path, depth_map_path, output_path)
