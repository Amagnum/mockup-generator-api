import { useState, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { generateMockup } from '../services/api';

const useMockupGenerator = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [sessionId, setSessionId] = useState(() => uuidv4());

  const resetSession = useCallback(() => {
    setSessionId(uuidv4());
    setResult(null);
  }, []);

  const createMockup = useCallback(async (files, params, advancedSettings) => {
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      
      // Add files
      formData.append('source_image', files.sourceImage);
      formData.append('source_mask', files.sourceMask);
      formData.append('source_depth', files.sourceDepth);
      formData.append('design_image', files.designImage);
      
      // Add parameters
      formData.append('color_code', params.colorCode);
      formData.append('location_x', params.locationX);
      formData.append('location_y', params.locationY);
      formData.append('scale_factor', params.scaleFactor);
      formData.append('shading_strength', params.shadingStrength);
      formData.append('color_mode', params.colorMode);
      formData.append('color_method', params.colorMethod);
      formData.append('session_id', sessionId);
      
      // Add advanced settings if using int-v2 method
      if (params.colorMethod === 'int-v2' && advancedSettings) {
        formData.append('color_config', JSON.stringify(advancedSettings));
      }
      
      const mockupResult = await generateMockup(formData);
      setResult(mockupResult);
      return mockupResult;
    } catch (err) {
      const errorMessage = err.detail || 'An error occurred while generating the mockup';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  return {
    loading,
    error,
    result,
    sessionId,
    createMockup,
    resetSession,
  };
};

export default useMockupGenerator; 