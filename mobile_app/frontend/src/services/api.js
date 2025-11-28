import axios from 'axios';

const API_BASE_URL = 'http://10.51.23.148:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout for large image uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('medSeg_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle response errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login
      localStorage.removeItem('medSeg_token');
      localStorage.removeItem('medSeg_user');
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);

export const authService = {
  login: async (credentials) => {
    const response = await api.post('/auth/login', credentials);
    return response.data;
  },
  register: async (userData) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },
};

export const modelService = {
  getModels: async () => {
    const response = await api.get('/models');
    return response.data;
  },
};

export const segmentationService = {
  segmentImage: async (imageFile, modelId, boundingBox, threshold = 0.5) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('model_id', modelId);
    formData.append('x1', boundingBox.x1.toString());
    formData.append('y1', boundingBox.y1.toString());
    formData.append('x2', boundingBox.x2.toString());
    formData.append('y2', boundingBox.y2.toString());
    formData.append('threshold', threshold.toString());

    const response = await api.post('/segment', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default api;