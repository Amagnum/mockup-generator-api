# T-Shirt Coloring Methods Documentation

This document provides detailed information about the different coloring methods available in the T-Shirt Mockup Generator API, with a focus on the advanced `int-v2` method and its configurable parameters.

## Available Coloring Methods

The API supports several coloring methods, each with different characteristics and use cases:

### 1. Standard Method

The default method that automatically selects between light and dark coloring approaches based on the target color's brightness.

- **Light Mode**: Uses HSV color space to preserve shading while applying light colors.
- **Dark Mode**: Uses multiply blending to better preserve shadows and details with dark colors.
- **Best for**: General purpose use with a wide range of colors.

### 2. Intrinsic Method

Uses intrinsic image decomposition to separate the image into albedo (base color) and shading components, then applies the new color while preserving the original shading details.

- **Best for**: Preserving fabric texture and natural shading.
- **Limitations**: May not handle very dark colors as well as the standard method.

### 3. Luminance-based Method (lum-v2)

Extracts the luminance from the original image and multiplies it with the target color to preserve shading and texture details.

- **Best for**: Clean, consistent coloring with good detail preservation.
- **Limitations**: May lose some subtle texture details compared to the intrinsic methods.

### 4. Enhanced Intrinsic Method (int-v2)

An advanced method that improves upon the original intrinsic decomposition by using a multi-scale approach to better separate texture from shading and applying adaptive color blending based on local texture complexity.

- **Best for**: High-quality mockups with excellent texture preservation and realistic shading.
- **Advantages**: Highly configurable for different fabric types and visual styles.
- **Use cases**: Professional mockups, detailed fabric rendering, and situations where texture preservation is critical.

## Enhanced Intrinsic Method (int-v2) Parameters

The `int-v2` method offers extensive customization through the following parameters:

### Scale and Blur Parameters

These parameters control how the algorithm analyzes and processes different levels of detail in the image:

#### `large_scale_blur` (default: 21)

Controls the kernel size for extracting large-scale shading information.

- **Higher values** (e.g., 31-41):
  - Create smoother, more uniform shading across the t-shirt
  - Reduce the influence of local fabric variations
  - Good for creating a cleaner, more uniform look

- **Lower values** (e.g., 11-15):
  - Preserve more of the original shading variations
  - Maintain more local contrast
  - Better for highlighting the natural contours of the t-shirt

- **Use cases**:
  - Higher for smooth fabrics like silk or jersey
  - Lower for fabrics with visible texture like cotton or linen

#### `medium_scale_blur` (default: 7)

Controls the kernel size for extracting medium-scale details like fabric patterns and folds.

- **Higher values** (e.g., 9-13):
  - Capture broader fabric patterns and larger folds
  - Reduce the visibility of smaller details
  - Create a smoother overall appearance

- **Lower values** (e.g., 3-5):
  - Focus on finer fabric weave patterns
  - Preserve more medium-scale details
  - Better for textured fabrics

- **Use cases**:
  - Higher for smooth, less textured fabrics
  - Lower for fabrics with visible weave patterns

#### `fine_scale_blur` (default: 3)

Controls the kernel size for extracting fine details like individual threads or micro-texture.

- **Higher values** (e.g., 5-7):
  - Reduce noise and very fine details
  - Create a cleaner, less grainy appearance
  - Good for smoother fabrics or when a cleaner look is desired

- **Lower values** (e.g., 1-2):
  - Preserve more micro-texture details
  - Maintain the finest fabric details
  - Better for highly detailed fabrics

- **Use cases**:
  - Higher for digital or print mockups where fine noise might be distracting
  - Lower for close-up mockups where fabric texture is important

### Weight Parameters

These parameters control how much each level of detail influences the final result:

#### `large_scale_weight` (default: 0.7)

Controls how much the large-scale shading influences the final result.

- **Higher values** (e.g., 0.8-1.0):
  - Create more pronounced overall shading
  - Enhance the 3D appearance of the t-shirt
  - Better for dramatic lighting or to emphasize body contours

- **Lower values** (e.g., 0.4-0.6):
  - Flatten the overall shading
  - Create a more uniform color application
  - Good for a cleaner, more graphic look

- **Use cases**:
  - Higher for realistic mockups with pronounced body contours
  - Lower for flat-lay mockups or when a more uniform appearance is desired

#### `medium_scale_weight` (default: 1.5)

Controls how much medium-scale details are emphasized.

- **Higher values** (e.g., 1.8-2.5):
  - Enhance fabric patterns and folds
  - Create more visible texture
  - Better for textured fabrics or to emphasize wrinkles

- **Lower values** (e.g., 0.8-1.2):
  - Create a smoother appearance
  - Reduce the visibility of fabric patterns
  - Good for smooth fabrics or a cleaner look

- **Use cases**:
  - Higher for casual t-shirts with natural wrinkles
  - Lower for premium apparel with a smoother appearance

#### `fine_scale_weight` (default: 2.0)

Controls how much fine texture details are emphasized.

- **Higher values** (e.g., 2.5-3.5):
  - Create more textured, fabric-like appearance
  - Enhance the visibility of individual threads and micro-texture
  - Better for highly detailed mockups

- **Lower values** (e.g., 1.0-1.5):
  - Create a smoother, less detailed result
  - Reduce the visibility of fine texture
  - Good for smoother fabrics or when a cleaner look is desired

- **Use cases**:
  - Higher for rough fabrics like canvas or heavy cotton
  - Lower for smooth fabrics like silk or jersey

### Shading Parameters

These parameters control the overall contrast and shadow depth:

#### `min_shading` (default: 0.05)

Sets the minimum brightness level in shadowed areas.

- **Higher values** (e.g., 0.1-0.2):
  - Prevent very dark shadows
  - Create a flatter, more uniform look
  - Good for light colors or when shadows should be subtle

- **Lower values** (e.g., 0.01-0.03):
  - Allow deeper shadows
  - Create more contrast
  - Better for dark colors or dramatic lighting

- **Use cases**:
  - Higher for pastel or light colors
  - Lower for dark colors or high-contrast mockups

#### `shading_boost` (default: 1.2)

Multiplier for the final shading map.

- **Higher values** (e.g., 1.3-1.5):
  - Increase overall contrast
  - Create more pronounced shadows and highlights
  - Better for dramatic lighting or to emphasize texture

- **Lower values** (e.g., 0.9-1.1):
  - Create a flatter, more uniform coloring
  - Reduce the contrast between shadows and highlights
  - Good for a cleaner, more graphic look

- **Use cases**:
  - Higher for realistic mockups with dramatic lighting
  - Lower for flat-lay mockups or when a more uniform appearance is desired

### Detail Preservation Parameters

These parameters control how much of the original fabric texture is preserved:

#### `base_detail_preservation` (default: 0.15)

Base amount of original image detail to preserve.

- **Higher values** (e.g., 0.2-0.3):
  - Retain more of the original fabric appearance
  - Preserve more of the original color variations
  - Better for highly textured fabrics

- **Lower values** (e.g., 0.05-0.1):
  - Apply more of the target color
  - Create a more uniform color application
  - Good for a cleaner, more graphic look

- **Use cases**:
  - Higher for vintage or distressed t-shirts
  - Lower for solid-color premium apparel

#### `texture_detail_weight` (default: 0.25)

How much additional detail to preserve in textured areas.

- **Higher values** (e.g., 0.3-0.4):
  - Preserve more detail in complex fabric areas
  - Create more visible texture in areas with high detail
  - Better for fabrics with varying texture

- **Lower values** (e.g., 0.1-0.2):
  - Create more uniform coloring across the fabric
  - Reduce the visibility of local texture variations
  - Good for a more consistent appearance

- **Use cases**:
  - Higher for natural fabrics with visible texture variations
  - Lower for synthetic fabrics with more uniform texture

#### `saturation_influence` (default: 0.3)

How much the target color's saturation reduces detail preservation.

- **Higher values** (e.g., 0.4-0.6):
  - Make saturated colors appear more uniform
  - Reduce texture visibility with vibrant colors
  - Good for bright, solid-color mockups

- **Lower values** (e.g., 0.1-0.2):
  - Preserve more texture even with saturated colors
  - Maintain fabric details regardless of color saturation
  - Better when texture preservation is critical

- **Use cases**:
  - Higher for graphic designs with vibrant colors
  - Lower for fashion mockups where fabric texture is important

## Recommended Configurations for Different Fabric Types

### Cotton T-shirts (Standard)
json
{
"large_scale_weight": 0.7,
"medium_scale_weight": 1.5,
"fine_scale_weight": 2.0,
"base_detail_preservation": 0.15,
"texture_detail_weight": 0.25
}

### Highly Textured Fabrics (Linen, Canvas, Heavy Cotton)
json
{
"large_scale_weight": 0.6,
"medium_scale_weight": 1.8,
"fine_scale_weight": 2.5,
"base_detail_preservation": 0.2,
"texture_detail_weight": 0.35,
"saturation_influence": 0.2
}

### Smooth Fabrics (Silk, Jersey, Polyester)
json
{
"large_scale_weight": 0.8,
"medium_scale_weight": 1.2,
"fine_scale_weight": 1.5,
"base_detail_preservation": 0.1,
"texture_detail_weight": 0.15,
"min_shading": 0.07
}


### Vintage or Distressed Look
json
{
"large_scale_weight": 0.6,
"medium_scale_weight": 1.7,
"fine_scale_weight": 2.8,
"base_detail_preservation": 0.25,
"texture_detail_weight": 0.4,
"saturation_influence": 0.15
}


### Dark Colors
json
{
"min_shading": 0.08,
"shading_boost": 1.3,
"saturation_influence": 0.4
}


### Light/Pastel Colors
json
{
"min_shading": 0.03,
"shading_boost": 1.1,
"saturation_influence": 0.25
}


## Technical Implementation Details

The `int-v2` method works by:

1. Decomposing the image into multiple frequency bands to separate different levels of detail
2. Analyzing local texture complexity to determine how much original detail to preserve
3. Applying adaptive color blending based on texture complexity and color saturation
4. Recombining the frequency bands with appropriate weights to create the final result

This multi-scale approach allows for much better preservation of fabric texture and natural shading compared to simpler coloring methods, resulting in more realistic and visually appealing mockups.

## Performance Considerations

The `int-v2` method is more computationally intensive than the standard methods due to the multiple decomposition steps and adaptive blending. For applications where processing speed is critical, consider:

- Using the standard method for preview images and `int-v2` for final renders
- Reducing the image resolution before applying the coloring
- Pre-computing and caching results for commonly used colors

## Troubleshooting

### Common Issues and Solutions

1. **Too much texture/noise in solid areas**
   - Increase `large_scale_blur` and `medium_scale_blur`
   - Decrease `fine_scale_weight`
   - Decrease `texture_detail_weight`

2. **Too flat/uniform appearance**
   - Increase `medium_scale_weight` and `fine_scale_weight`
   - Decrease `large_scale_weight`
   - Increase `base_detail_preservation`

3. **Shadows too dark**
   - Increase `min_shading`
   - Decrease `shading_boost`

4. **Insufficient contrast**
   - Increase `shading_boost`
   - Decrease `min_shading`
   - Increase weights for all scale levels