import React, { useState } from 'react';
import { Box, Typography, TextField, InputAdornment, Paper, Grid, Tooltip } from '@mui/material';
import { ChromePicker, SketchPicker } from 'react-color';

// Preset colors
const presetColors = [
  "#2f3424", // Dark olive
  "#474747", // Dark gray
  "#f7e471", // Light yellow
  "#8f87be", // Lavender
  "#ffcef3", // Light pink
];

const ColorPicker = ({ color, onChange }) => {
  const [displayColorPicker, setDisplayColorPicker] = useState(false);
  const [inputValue, setInputValue] = useState(color);

  const handleClick = () => {
    setDisplayColorPicker(!displayColorPicker);
  };

  const handleClose = () => {
    setDisplayColorPicker(false);
  };

  const handleChange = (newColor) => {
    onChange(newColor.hex);
    setInputValue(newColor.hex);
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    // Validate hex color format
    const isValidHex = /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(inputValue);
    if (isValidHex) {
      onChange(inputValue);
    } else {
      // Reset to current color if invalid
      setInputValue(color);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleInputBlur();
    }
  };

  return (
    <Box>
      <Typography variant="subtitle1" gutterBottom>
        Product Color
      </Typography>
      <Paper elevation={0} sx={{ p: 1, mb: 2 }}>
        <Grid container spacing={1} sx={{ mb: 2 }}>
          {presetColors.map((presetColor) => (
            <Grid item key={presetColor}>
              <Tooltip title={presetColor}>
                <Box
                  onClick={() => onChange(presetColor)}
                  sx={{
                    width: 36,
                    height: 36,
                    borderRadius: "50%",
                    backgroundColor: presetColor,
                    cursor: "pointer",
                    border: color === presetColor ? "2px solid #000" : "1px solid #ddd",
                    transition: "transform 0.2s",
                    "&:hover": {
                      transform: "scale(1.1)",
                    },
                  }}
                />
              </Tooltip>
            </Grid>
          ))}
        </Grid>
        <SketchPicker
          color={color}
          onChange={(colorObj) => onChange(colorObj.hex)}
          disableAlpha={true}
          width="100%"
        />
      </Paper>
      
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Box
          sx={{
            width: 36,
            height: 36,
            borderRadius: 1,
            backgroundColor: color,
            cursor: 'pointer',
            border: '1px solid #ccc',
            mr: 2
          }}
          onClick={handleClick}
        />
        
        <TextField
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyDown={handleKeyDown}
          size="small"
          InputProps={{
            startAdornment: <InputAdornment position="start">#</InputAdornment>,
          }}
          sx={{ width: '140px' }}
          placeholder="Hex color"
        />
      </Box>
      
      {displayColorPicker && (
        <Box sx={{ position: 'absolute', zIndex: 2 }}>
          <Box
            sx={{
              position: 'fixed',
              top: 0,
              right: 0,
              bottom: 0,
              left: 0,
            }}
            onClick={handleClose}
          />
          <ChromePicker color={color} onChange={handleChange} />
        </Box>
      )}
    </Box>
  );
};

export default ColorPicker; 