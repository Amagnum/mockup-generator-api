import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const generateMockup = async (formData) => {
  try {
    const response = await api.post('/mockups/generate', formData);
    return response.data;
  } catch (error) {
    console.error('Error generating mockup:', error);
    throw error.response?.data || error.message;
  }
};

export const downloadMockup = (mockupId) => {
  // C:\Users\admin\OneDrive\Desktop\Tellotale\DriveAutomation\bulk-mockup-api\output\48ee2d68-0efc-4f9d-8ca7-6a20e3ea3b38.png
  // return `C:\\Users\\admin\\OneDrive\\Desktop\\Tellotale\\DriveAutomation\\bulk-mockup-api\\output\\${mockupId}.png`;
  return `${API_URL}/mockups/${mockupId}/download`;
};

export const deleteMockup = async (mockupId) => {
  try {
    const response = await api.delete(`/mockups/${mockupId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting mockup:', error);
    throw error.response?.data || error.message;
  }
};

export default api; 