import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  Paper,
  Grid,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Divider,
  IconButton,
  Tooltip,
  Card,
  CardContent,
  Stack,
  Chip,
} from "@mui/material";
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import ImageUploader from "./ImageUploader";
import ColorPicker from "./ColorPicker";
import DesignControls from "./DesignControls";
import MockupPreview from "./MockupPreview";
import AdvancedSettings from "./AdvancedSettings";
import { generateMockup, downloadMockup } from "../services/api";
import { v4 as uuidv4 } from "uuid";

const MockupGenerator = () => {
  // Define default values for mockup parameters and advanced settings
  const defaultMockupParams = {
    colorCode: "#3f51b5",
    locationX: 400,
    locationY: 400,
    scaleFactor: 0.5,
    shadingStrength: 0.7,
    colorMode: "auto",
    colorMethod: "standard",
  };

  const defaultAdvancedSettings = {
    largeScaleBlur: 21,
    mediumScaleBlur: 7,
    fineScaleBlur: 3,
    largeScaleWeight: 0.7,
    mediumScaleWeight: 1.5,
    fineScaleWeight: 2.0,
    minShading: 0.05,
    shadingBoost: 1.2,
    baseDetailPreservation: 0.15,
    textureDetailWeight: 0.25,
    saturationInfluence: 0.3,
  };

  const [sessionId, setSessionId] = useState(null);
  const [files, setFiles] = useState({
    sourceImage: null,
    sourceMask: null,
    sourceDepth: null,
  });
  
  // New state for multiple design images and colors
  const [designImages, setDesignImages] = useState([{ id: uuidv4(), file: null }]);
  const [colorCodes, setColorCodes] = useState([{ id: uuidv4(), value: "#3f51b5" }]);
  
  const [mockupParams, setMockupParams] = useState({...defaultMockupParams});
  const [advancedSettings, setAdvancedSettings] = useState({...defaultAdvancedSettings});
  const [mockupResults, setMockupResults] = useState({});
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState(null);

  useEffect(() => {
    // Generate a session ID when the component mounts
    setSessionId(uuidv4());
  }, []);

  const handleFileChange = (type, file) => {
    if (type === "designImage") {
      // Update the first design image in the array
      const updatedDesignImages = [...designImages];
      updatedDesignImages[0].file = file;
      setDesignImages(updatedDesignImages);
      
      // If all required files are present, trigger auto-update for all combinations
      if (autoUpdate && files.sourceImage && files.sourceMask && files.sourceDepth && file) {
        generateAllMockups();
      }
    } else {
      setFiles((prevFiles) => {
        const newFiles = {
          ...prevFiles,
          [type]: file,
        };

        // If all required files are present, trigger auto-update for all combinations
        if (
          autoUpdate &&
          newFiles.sourceImage &&
          newFiles.sourceMask &&
          newFiles.sourceDepth &&
          designImages.some(design => design.file)
        ) {
          generateAllMockups();
        }

        return newFiles;
      });
    }
  };

  const handleParamChange = (param, value) => {
    // Validate numeric inputs
    if (
      ["locationX", "locationY", "scaleFactor", "shadingStrength"].includes(
        param
      )
    ) {
      // Convert to number and validate
      const numValue = Number(value);
      if (isNaN(numValue)) return;

      // Apply min/max constraints if needed
      if (param === "scaleFactor" && (numValue < 0.1 || numValue > 2)) return;
      if (param === "shadingStrength" && (numValue < 0 || numValue > 1)) return;

      value = numValue;
    }

    setMockupParams((prevParams) => {
      const newParams = {
        ...prevParams,
        [param]: value,
      };

      // Debounce parameter changes to avoid too many API calls
      if (autoUpdate && allFilesPresent()) {
        debounceAllUpdates();
      }

      return newParams;
    });
  };

  const handleAdvancedSettingChange = (setting, value) => {
    // Convert to number and validate
    const numValue = Number(value);
    if (isNaN(numValue)) return;

    setAdvancedSettings((prevSettings) => {
      const newSettings = {
        ...prevSettings,
        [setting]: numValue,
      };

      // Debounce advanced settings changes
      if (
        autoUpdate &&
        allFilesPresent() &&
        mockupParams.colorMethod === "int-v2"
      ) {
        debounceAllUpdates();
      }

      return newSettings;
    });
  };

  const allFilesPresent = () => {
    return (
      files.sourceImage &&
      files.sourceMask &&
      files.sourceDepth &&
      designImages.some(design => design.file)
    );
  };

  // Debounce function to prevent too many API calls
  const debounceAllUpdates = useCallback(() => {
    const now = Date.now();

    // Only update if it's been more than 1 second since the last update
    if (!lastUpdateTime || now - lastUpdateTime > 1000) {
      generateAllMockups();
      setLastUpdateTime(now);
    } else {
      // Schedule an update after the debounce period
      const timeSinceLastUpdate = now - lastUpdateTime;
      const timeToWait = Math.max(0, 1000 - timeSinceLastUpdate);

      setTimeout(() => {
        generateAllMockups();
        setLastUpdateTime(Date.now());
      }, timeToWait);
    }
  }, [lastUpdateTime]);

  // Generate mockups for all design and color combinations
  const generateAllMockups = () => {
    designImages.forEach(design => {
      if (design.file) {
        colorCodes.forEach(color => {
          const combinationId = `${design.id}-${color.id}`;
          triggerUpdate(design.file, color.value, combinationId);
        });
      }
    });
  };

  const triggerUpdate = async (designFile, colorCode, combinationId) => {
    if (loading[combinationId]) return; // Don't trigger if already loading

    setLoading(prev => ({ ...prev, [combinationId]: true }));
    setError(null);

    try {
      const formData = new FormData();

      // Add files
      formData.append("source_image", files.sourceImage);
      formData.append("source_mask", files.sourceMask);
      formData.append("source_depth", files.sourceDepth);
      formData.append("design_image", designFile);

      // Add parameters
      formData.append("color_code", colorCode);
      formData.append("location_x", mockupParams.locationX);
      formData.append("location_y", mockupParams.locationY);
      formData.append("scale_factor", mockupParams.scaleFactor);
      formData.append("shading_strength", mockupParams.shadingStrength);
      formData.append("color_mode", mockupParams.colorMode);
      formData.append("color_method", mockupParams.colorMethod);
      
      // Generate a unique session ID for each combination to allow parallel processing
      const uniqueSessionId = `${sessionId}-${combinationId}`;
      formData.append("session_id", uniqueSessionId);

      // Add advanced settings if using int-v2 method
      if (mockupParams.colorMethod === "int-v2") {
        // Convert settings to the correct format from camelCase to snake_case
        const convertedSettings = Object.fromEntries(
          Object.entries(advancedSettings).map(([key, value]) => [
            key.replace(/([A-Z])/g, "_$1").toLowerCase(),
            value,
          ])
        );
        formData.append("color_config", JSON.stringify(convertedSettings));
      }

      const result = await generateMockup(formData);
      setMockupResults(prev => ({ ...prev, [combinationId]: result }));
    } catch (err) {
      setError(err.detail || "An error occurred while generating the mockup");
    } finally {
      setLoading(prev => ({ ...prev, [combinationId]: false }));
    }
  };

  const handleGenerateMockup = () => {
    generateAllMockups();
  };

  const handleDownload = (mockupId) => {
    if (mockupId) {
      window.open(downloadMockup(mockupId), "_blank");
    }
  };

  const toggleAutoUpdate = () => {
    setAutoUpdate(!autoUpdate);
  };

  const resetMockupParams = () => {
    setMockupParams({...defaultMockupParams});
    if (autoUpdate && allFilesPresent()) {
      debounceAllUpdates();
    }
  };

  const resetAdvancedSettings = () => {
    setAdvancedSettings({...defaultAdvancedSettings});
    if (autoUpdate && allFilesPresent() && mockupParams.colorMethod === "int-v2") {
      debounceAllUpdates();
    }
  };

  const resetAllFiles = () => {
    setFiles({
      sourceImage: null,
      sourceMask: null,
      sourceDepth: null,
    });
    setDesignImages([{ id: uuidv4(), file: null }]);
    setMockupResults({});
  };

  // Add a new design image
  const addDesignImage = () => {
    setDesignImages(prev => [...prev, { id: uuidv4(), file: null }]);
  };

  // Remove a design image
  const removeDesignImage = (id) => {
    if (designImages.length <= 1) return; // Keep at least one design image
    
    setDesignImages(prev => prev.filter(design => design.id !== id));
    
    // Remove associated mockup results
    const updatedResults = { ...mockupResults };
    colorCodes.forEach(color => {
      const combinationId = `${id}-${color.id}`;
      delete updatedResults[combinationId];
    });
    setMockupResults(updatedResults);
  };

  // Add a new color
  const addColor = () => {
    setColorCodes(prev => [...prev, { id: uuidv4(), value: "#" + Math.floor(Math.random()*16777215).toString(16) }]);
  };

  // Remove a color
  const removeColor = (id) => {
    if (colorCodes.length <= 1) return; // Keep at least one color
    
    setColorCodes(prev => prev.filter(color => color.id !== id));
    
    // Remove associated mockup results
    const updatedResults = { ...mockupResults };
    designImages.forEach(design => {
      const combinationId = `${design.id}-${id}`;
      delete updatedResults[combinationId];
    });
    setMockupResults(updatedResults);
  };

  // Update a specific color
  const updateColor = (id, newColor) => {
    setColorCodes(prev => 
      prev.map(color => 
        color.id === id ? { ...color, value: newColor } : color
      )
    );
    
    // If auto-update is on, regenerate mockups for this color
    if (autoUpdate && allFilesPresent()) {
      designImages.forEach(design => {
        if (design.file) {
          const combinationId = `${design.id}-${id}`;
          triggerUpdate(design.file, newColor, combinationId);
        }
      });
    }
  };

  // Update a specific design image
  const updateDesignImage = (id, file) => {
    setDesignImages(prev => 
      prev.map(design => 
        design.id === id ? { ...design, file } : design
      )
    );
    
    // If auto-update is on, regenerate mockups for this design
    if (autoUpdate && allFilesPresent() && file) {
      colorCodes.forEach(color => {
        const combinationId = `${id}-${color.id}`;
        triggerUpdate(file, color.value, combinationId);
      });
    }
  };

  return (
    <Box sx={{ width: "100%" }}>
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Left Column - Controls */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h6">Upload Base Images</Typography>
              <Tooltip title="Reset all images">
                <IconButton onClick={resetAllFiles} color="primary" size="small">
                  <RestartAltIcon />
                </IconButton>
              </Tooltip>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <ImageUploader 
              files={{
                sourceImage: files.sourceImage,
                sourceMask: files.sourceMask,
                sourceDepth: files.sourceDepth,
                designImage: designImages[0]?.file || null
              }} 
              onFileChange={handleFileChange} 
            />
          </Paper>

          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h6">Design Images</Typography>
              <Tooltip title="Add design image">
                <IconButton onClick={addDesignImage} color="primary" size="small">
                  <AddIcon />
                </IconButton>
              </Tooltip>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              {designImages.map((design, index) => (
                <Grid item xs={12} sm={6} key={design.id}>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                        <Typography variant="subtitle2">Design {index + 1}</Typography>
                        {designImages.length > 1 && (
                          <IconButton 
                            onClick={() => removeDesignImage(design.id)} 
                            color="error" 
                            size="small"
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        )}
                      </Box>
                      <Box 
                        sx={{ 
                          border: '1px dashed #ccc', 
                          borderRadius: 1, 
                          p: 1, 
                          minHeight: '100px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          position: 'relative',
                          overflow: 'hidden',
                          backgroundColor: '#f5f5f5'
                        }}
                      >
                        {design.file ? (
                          <>
                            <img 
                              src={URL.createObjectURL(design.file)} 
                              alt={`Design ${index + 1}`} 
                              style={{ maxWidth: '100%', maxHeight: '100px' }} 
                            />
                            <IconButton 
                              sx={{ position: 'absolute', top: 0, right: 0, backgroundColor: 'rgba(255,255,255,0.7)' }}
                              onClick={() => updateDesignImage(design.id, null)}
                              size="small"
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </>
                        ) : (
                          <Button 
                            variant="text" 
                            component="label"
                            size="small"
                          >
                            Upload Design
                            <input
                              type="file"
                              hidden
                              accept="image/*"
                              onChange={(e) => {
                                if (e.target.files && e.target.files[0]) {
                                  updateDesignImage(design.id, e.target.files[0]);
                                }
                              }}
                            />
                          </Button>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>

          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h6">Colors</Typography>
              <Tooltip title="Add color">
                <IconButton onClick={addColor} color="primary" size="small">
                  <AddIcon />
                </IconButton>
              </Tooltip>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              {colorCodes.map((color, index) => (
                <Grid item xs={12} sm={6} md={4} key={color.id}>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                        <Typography variant="subtitle2">Color {index + 1}</Typography>
                        {colorCodes.length > 1 && (
                          <IconButton 
                            onClick={() => removeColor(color.id)} 
                            color="error" 
                            size="small"
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        )}
                      </Box>
                      <ColorPicker
                        color={color.value}
                        onChange={(newColor) => updateColor(color.id, newColor)}
                      />
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>

          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h6">Design Settings</Typography>
              <Tooltip title="Reset to default settings">
                <IconButton onClick={resetMockupParams} color="primary" size="small">
                  <RestartAltIcon />
                </IconButton>
              </Tooltip>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <DesignControls
              params={mockupParams}
              onChange={handleParamChange}
            />
          </Paper>

          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h6">Advanced Settings</Typography>
              <Tooltip title="Reset to default advanced settings">
                <IconButton onClick={resetAdvancedSettings} color="primary" size="small">
                  <RestartAltIcon />
                </IconButton>
              </Tooltip>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <AdvancedSettings
              colorMethod={mockupParams.colorMethod}
              settings={advancedSettings}
              onChange={handleAdvancedSettingChange}
              onMethodChange={(method) =>
                handleParamChange("colorMethod", method)
              }
            />
          </Paper>
        </Grid>

        {/* Right Column - Preview */}
        <Grid item xs={12} md={6}>
          <Paper
            sx={{
              p: 3,
              height: "100%",
              maxHeight: "calc(100vh - 32px)", // Account for margin/padding
              overflow: "auto", // Enable scrolling
              position: "sticky",
              top: 0,
              zIndex: 1000,
            }}
          >
            <Box
              sx={{ display: "flex", justifyContent: "space-between", mb: 3 }}
            >
              <Button
                variant={autoUpdate ? "outlined" : "contained"}
                color={autoUpdate ? "success" : "primary"}
                onClick={toggleAutoUpdate}
                startIcon={autoUpdate ? null : <RestartAltIcon />}
              >
                {autoUpdate ? "Auto-Update ON" : "Auto-Update OFF"}
              </Button>

              <Button
                variant="contained"
                onClick={handleGenerateMockup}
                disabled={Object.values(loading).some(Boolean) || !allFilesPresent()}
              >
                {Object.values(loading).some(Boolean) ? "Generating..." : "Generate All Mockups"}
              </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Mockup Previews
            </Typography>

            {designImages.some(design => design.file) ? (
              designImages.map(design => {
                if (!design.file) return null;
                
                return (
                  <Box key={design.id} sx={{ mb: 4 }}>
                    <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
                      Design {designImages.findIndex(d => d.id === design.id) + 1}
                    </Typography>
                    <Grid container spacing={2}>
                      {colorCodes.map(color => {
                        const combinationId = `${design.id}-${color.id}`;
                        const result = mockupResults[combinationId];
                        const isLoading = loading[combinationId];
                        
                        return (
                          <Grid item xs={12} sm={6} key={combinationId}>
                            <Card variant="outlined">
                              <CardContent sx={{ p: 1 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                  <Box 
                                    sx={{ 
                                      width: 20, 
                                      height: 20, 
                                      borderRadius: '50%', 
                                      backgroundColor: color.value,
                                      border: '1px solid #ddd',
                                      mr: 1
                                    }} 
                                  />
                                  <Typography variant="caption">
                                    {color.value}
                                  </Typography>
                                </Box>
                                
                                {isLoading ? (
                                  <Box
                                    sx={{
                                      display: "flex",
                                      justifyContent: "center",
                                      alignItems: "center",
                                      height: 200,
                                    }}
                                  >
                                    <CircularProgress size={30} />
                                  </Box>
                                ) : result ? (
                                  <Box>
                                    <img 
                                      src={downloadMockup(result.mockup_id)} 
                                      alt={`Mockup with ${color.value}`}
                                      style={{ width: '100%', height: 'auto' }}
                                    />
                                    <Button 
                                      variant="outlined" 
                                      size="small" 
                                      fullWidth 
                                      onClick={() => handleDownload(result.mockup_id)}
                                      sx={{ mt: 1 }}
                                    >
                                      Download
                                    </Button>
                                  </Box>
                                ) : (
                                  <Box
                                    sx={{
                                      display: "flex",
                                      flexDirection: "column",
                                      justifyContent: "center",
                                      alignItems: "center",
                                      height: 200,
                                      border: "1px dashed #ccc",
                                      borderRadius: 1,
                                    }}
                                  >
                                    <Typography variant="caption" color="text.secondary">
                                      Click 'Generate All Mockups' to preview
                                    </Typography>
                                  </Box>
                                )}
                              </CardContent>
                            </Card>
                          </Grid>
                        );
                      })}
                    </Grid>
                  </Box>
                );
              })
            ) : (
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                  minHeight: "400px",
                  border: "1px dashed #ccc",
                  borderRadius: 2,
                  p: 3,
                }}
              >
                <Typography
                  variant="body1"
                  color="text.secondary"
                  align="center"
                >
                  {allFilesPresent()
                    ? "Click 'Generate All Mockups' to see the previews"
                    : "Upload all required images to generate previews"}
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MockupGenerator;
