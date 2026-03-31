import axios from 'axios';
import Constants from 'expo-constants';
import { BackendStatusResponse } from '../types';

// Extract the exact IP address running the Expo Dev Server
// This completely fixes "localhost vs IP" issues when testing on actual iPhones/Androids
const hostUri = Constants.expoConfig?.hostUri;
const ipAddress = hostUri ? hostUri.split(':')[0] : '127.0.0.1';
const BASE_URL = `http://${ipAddress}:8000`;

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
    console.warn('API Error:', error.message);
    return Promise.reject(error);
  }
);

export const fetchStatus = async (): Promise<BackendStatusResponse> => {
  const { data } = await apiClient.get<BackendStatusResponse>('/status');
  return data;
};

export const sendEyeEvent = async (name: 'look_left' | 'look_right' | 'short_blink' | 'long_blink'): Promise<any> => {
  const { data } = await apiClient.post('/event', {
    event_type: 'eye',
    name,
  });
  return data;
};
