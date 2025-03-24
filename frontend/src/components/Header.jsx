import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import CheckroomIcon from '@mui/icons-material/Checkroom';
import Box from '@mui/material/Box';

const Header = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <CheckroomIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          T-Shirt Mockup Generator
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 