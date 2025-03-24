import React from 'react';
import { Box, Typography, Grid, Paper } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const ImageUploader = ({ files, onFileChange }) => {
  const useCreateDropzone = (type, label, accept) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
      accept: accept,
      maxFiles: 1,
      onDrop: (acceptedFiles) => {
        if (acceptedFiles.length > 0) {
          onFileChange(type, acceptedFiles[0]);
        }
      },
    });

    return (
      <Grid item xs={12} sm={6}>
        <Paper
          {...getRootProps()}
          sx={{
            p: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: 120,
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            backgroundColor: isDragActive ? 'rgba(63, 81, 181, 0.05)' : 'background.paper',
            cursor: 'pointer',
            '&:hover': {
              borderColor: 'primary.main',
              backgroundColor: 'rgba(63, 81, 181, 0.05)',
            },
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon color="primary" sx={{ fontSize: 30, mb: 1 }} />
          <Typography variant="body2" align="center" gutterBottom>
            {label}
          </Typography>
          {files[type] ? (
            <Typography variant="caption" color="text.secondary" noWrap sx={{ maxWidth: '100%' }}>
              {files[type].name}
            </Typography>
          ) : (
            <Typography variant="caption" color="text.secondary">
              Drag & drop or click
            </Typography>
          )}
        </Paper>
      </Grid>
    );
  };

  return (
    <Grid container spacing={2}>
      {useCreateDropzone('sourceImage', 'Source Image', { 'image/*': ['.jpeg', '.jpg', '.png'] })}
      {useCreateDropzone('sourceMask', 'T-shirt Mask', { 'image/*': ['.jpeg', '.jpg', '.png'] })}
      {useCreateDropzone('sourceDepth', 'Depth Map', { 'image/*': ['.jpeg', '.jpg', '.png'] })}
      {useCreateDropzone('designImage', 'Design Image', { 'image/*': ['.jpeg', '.jpg', '.png'] })}
    </Grid>
  );
};

export default ImageUploader; 