import React from 'react';
import { Box, Typography, Button, Card, CardMedia, CardContent, CardActions } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import { downloadMockup } from '../services/api';

const MockupPreview = ({ result, onDownload }) => {
  const [scale, setScale] = React.useState(1);
  const [position, setPosition] = React.useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = React.useState(false);
  const [dragStart, setDragStart] = React.useState({ x: 0, y: 0 });
  const [mockupUrl, setMockupUrl] = React.useState(null);

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY * -0.01;
    const newScale = Math.min(Math.max(scale + delta, 0.5), 3);
    setScale(newScale);
  };

  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    
    const newX = e.clientX - dragStart.x;
    const newY = e.clientY - dragStart.y;
    setPosition({ x: newX, y: newY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Prevent scrolling when hovering over the preview
  React.useEffect(() => {
    const preventDefault = (e) => e.preventDefault();
    const previewElement = document.getElementById('mockup-preview');
    
    if (previewElement) {
      previewElement.addEventListener('touchmove', preventDefault, { passive: false });
      previewElement.addEventListener('wheel', preventDefault, { passive: false });
    }

    return () => {
      if (previewElement) {
        previewElement.removeEventListener('touchmove', preventDefault);
        previewElement.removeEventListener('wheel', preventDefault);
      }
    };
  }, []);

  // Load the mockup URL once when the result changes
  React.useEffect(() => {
    if (result && result.mockup_id) {
      // Get the URL without adding a timestamp
      const url = downloadMockup(result.mockup_id);
      setMockupUrl(url);
    }
  }, [result]);

  if (!result) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <Typography variant="body1" color="text.secondary">
          No mockup generated yet
        </Typography>
      </Box>
    );
  }

  console.log(mockupUrl, result);

  return (
    <Box>
      <Card sx={{ maxWidth: '100%', mb: 3 }}>
        <Box
          id="mockup-preview"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          sx={{
            overflow: "hidden",
            cursor: isDragging ? "grabbing" : "grab",
            userSelect: "none",
          }}
        >
          {mockupUrl}
          <CardMedia
            component="img"
            image={mockupUrl}
            alt="T-shirt Mockup"
            sx={{
              maxHeight: "500px",
              objectFit: "contain",
              backgroundColor: "#f5f5f5",
              transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
              transition: isDragging ? "none" : "transform 0.1s ease-in-out",
            }}
          />
        </Box>
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            Mockup ID: {result.mockup_id}
          </Typography>
        </CardContent>
        <CardActions>
          <Button 
            variant="contained" 
            startIcon={<DownloadIcon />}
            onClick={onDownload}
            fullWidth
          >
            Download Mockup
          </Button>
        </CardActions>
      </Card>
    </Box>
  );
};

export default MockupPreview; 