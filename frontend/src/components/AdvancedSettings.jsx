import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Accordion, 
  AccordionSummary, 
  AccordionDetails, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Slider, 
  Grid, 
  TextField
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const AdvancedSettings = ({ colorMethod, settings, onChange, onMethodChange }) => {
  const [expanded, setExpanded] = useState(false);

  const handleMethodChange = (event) => {
    onMethodChange(event.target.value);
  };

  const handleSliderChange = (setting) => (_, value) => {
    onChange(setting, value);
  };

  const handleInputChange = (setting) => (event) => {
    onChange(setting, event.target.value);
  };

  const renderSliderWithInput = (label, setting, min, max, step) => (
    <Grid item xs={12}>
      <Typography variant="body2">{label}</Typography>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Slider
          value={settings[setting]}
          onChange={handleSliderChange(setting)}
          min={min}
          max={max}
          step={step}
          sx={{ mr: 2, flexGrow: 1 }}
          disabled={colorMethod !== 'int-v2'}
        />
        <TextField
          value={settings[setting]}
          onChange={handleInputChange(setting)}
          type="number"
          size="small"
          sx={{ width: '80px' }}
          inputProps={{ min, max, step }}
          disabled={colorMethod !== 'int-v2'}
        />
      </Box>
    </Grid>
  );

  return (
    <Box>
      <Typography variant="subtitle1" gutterBottom>
        Advanced Settings
      </Typography>
      
      <FormControl fullWidth size="small" sx={{ mb: 2 }}>
        <InputLabel>Color Method</InputLabel>
        <Select
          value={colorMethod}
          label="Color Method"
          onChange={handleMethodChange}
        >
          <MenuItem value="standard">Standard</MenuItem>
          <MenuItem value="int-v1">Intelligent V1</MenuItem>
          <MenuItem value="int-v2">Intelligent V2 (Advanced)</MenuItem>
        </Select>
      </FormControl>
      
      <Accordion 
        expanded={expanded || colorMethod === 'int-v2'} 
        onChange={() => setExpanded(!expanded)}
        disabled={colorMethod !== 'int-v2'}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Advanced Color Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {renderSliderWithInput('Large Scale Blur', 'largeScaleBlur', 1, 51, 2)}
            {renderSliderWithInput('Medium Scale Blur', 'mediumScaleBlur', 1, 21, 2)}
            {renderSliderWithInput('Fine Scale Blur', 'fineScaleBlur', 1, 11, 2)}
            {renderSliderWithInput('Large Scale Weight', 'largeScaleWeight', 0, 3, 0.1)}
            {renderSliderWithInput('Medium Scale Weight', 'mediumScaleWeight', 0, 3, 0.1)}
            {renderSliderWithInput('Fine Scale Weight', 'fineScaleWeight', 0, 3, 0.1)}
            {renderSliderWithInput('Min Shading', 'minShading', 0, 0.5, 0.01)}
            {renderSliderWithInput('Shading Boost', 'shadingBoost', 0.5, 3, 0.1)}
            {renderSliderWithInput('Base Detail Preservation', 'baseDetailPreservation', 0, 1, 0.01)}
            {renderSliderWithInput('Texture Detail Weight', 'textureDetailWeight', 0, 1, 0.01)}
            {renderSliderWithInput('Saturation Influence', 'saturationInfluence', 0, 1, 0.01)}
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default AdvancedSettings; 