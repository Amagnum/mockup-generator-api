import React from 'react';
import { Box, Typography, Slider, Grid, TextField, FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const DesignControls = ({ params, onChange }) => {
  const handleSliderChange = (param) => (_, value) => {
    onChange(param, value);
  };

  const handleInputChange = (param) => (event) => {
    onChange(param, event.target.value);
  };

  const handleSelectChange = (event) => {
    onChange('colorMode', event.target.value);
  };

  return (
    <Box>
      <Typography variant="subtitle1" gutterBottom>
        Design Controls
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Typography variant="body2">Position X</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Slider
              value={params.locationX}
              onChange={handleSliderChange('locationX')}
              min={0}
              max={1000}
              step={1}
              sx={{ mr: 2, flexGrow: 1 }}
            />
            <TextField
              value={params.locationX}
              onChange={handleInputChange('locationX')}
              type="number"
              size="small"
              sx={{ width: '80px' }}
              inputProps={{ min: 0, max: 1000 }}
            />
          </Box>
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="body2">Position Y</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Slider
              value={params.locationY}
              onChange={handleSliderChange('locationY')}
              min={0}
              max={1000}
              step={1}
              sx={{ mr: 2, flexGrow: 1 }}
            />
            <TextField
              value={params.locationY}
              onChange={handleInputChange('locationY')}
              type="number"
              size="small"
              sx={{ width: '80px' }}
              inputProps={{ min: 0, max: 1000 }}
            />
          </Box>
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="body2">Scale Factor</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Slider
              value={params.scaleFactor}
              onChange={handleSliderChange('scaleFactor')}
              min={0.1}
              max={2}
              step={0.01}
              sx={{ mr: 2, flexGrow: 1 }}
            />
            <TextField
              value={params.scaleFactor}
              onChange={handleInputChange('scaleFactor')}
              type="number"
              size="small"
              sx={{ width: '80px' }}
              inputProps={{ min: 0.1, max: 2, step: 0.01 }}
            />
          </Box>
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="body2">Shading Strength</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Slider
              value={params.shadingStrength}
              onChange={handleSliderChange('shadingStrength')}
              min={0}
              max={1}
              step={0.01}
              sx={{ mr: 2, flexGrow: 1 }}
            />
            <TextField
              value={params.shadingStrength}
              onChange={handleInputChange('shadingStrength')}
              type="number"
              size="small"
              sx={{ width: '80px' }}
              inputProps={{ min: 0, max: 1, step: 0.01 }}
            />
          </Box>
        </Grid>
        
        <Grid item xs={12}>
          <FormControl fullWidth size="small">
            <InputLabel>Color Mode</InputLabel>
            <Select
              value={params.colorMode}
              label="Color Mode"
              onChange={handleSelectChange}
            >
              <MenuItem value="auto">Auto</MenuItem>
              <MenuItem value="preserve">Preserve Original</MenuItem>
              <MenuItem value="replace">Replace Color</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DesignControls; 