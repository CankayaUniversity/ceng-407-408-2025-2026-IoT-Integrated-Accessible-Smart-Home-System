import axios from 'axios';

// This is a mock API Client setup to prepare for FastAPI integration.
const BASE_URL = 'http://raspberrypi5.local:8000/api/v1';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Mock interceptors for auth / error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle error
    console.warn('API Error:', error.message);
    return Promise.reject(error);
  }
);
