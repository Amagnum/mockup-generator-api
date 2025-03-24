/**
 * Converts a hex color code to RGB values
 * @param {string} hex - Hex color code (e.g., #FF0000)
 * @returns {Object} RGB values as {r, g, b}
 */
export const hexToRgb = (hex) => {
  // Remove # if present
  hex = hex.replace('#', '');
  
  // Parse the hex values
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  
  return { r, g, b };
};

/**
 * Determines if a color is light or dark
 * @param {string} hexColor - Hex color code
 * @returns {boolean} True if the color is light, false if dark
 */
export const isLightColor = (hexColor) => {
  const { r, g, b } = hexToRgb(hexColor);
  
  // Calculate perceived brightness using the formula:
  // (0.299*R + 0.587*G + 0.114*B)
  const brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  
  // If brightness is greater than 0.5, color is considered light
  return brightness > 0.5;
};

/**
 * Creates a preset configuration for different fabric types
 * @param {string} fabricType - Type of fabric ('cotton', 'linen', 'silk', etc.)
 * @returns {Object} Configuration object for the selected fabric type
 */
export const getFabricPreset = (fabricType) => {
  switch (fabricType.toLowerCase()) {
    case 'cotton':
      return {
        largeScaleWeight: 0.7,
        mediumScaleWeight: 1.5,
        fineScaleWeight: 2.0,
        baseDetailPreservation: 0.15,
        textureDetailWeight: 0.25
      };
    case 'linen':
      return {
        largeScaleWeight: 0.6,
        mediumScaleWeight: 1.8,
        fineScaleWeight: 2.5,
        baseDetailPreservation: 0.2,
        textureDetailWeight: 0.35
      };
    case 'silk':
      return {
        largeScaleWeight: 0.8,
        mediumScaleWeight: 1.2,
        fineScaleWeight: 1.5,
        baseDetailPreservation: 0.1,
        textureDetailWeight: 0.15
      };
    default:
      return {
        largeScaleWeight: 0.7,
        mediumScaleWeight: 1.5,
        fineScaleWeight: 2.0,
        baseDetailPreservation: 0.15,
        textureDetailWeight: 0.25
      };
  }
}; 