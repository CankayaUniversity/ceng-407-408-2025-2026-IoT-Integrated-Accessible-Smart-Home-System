import axios from 'axios';
import {
  BackendStatusResponse,
  MappingsResponse,
  VisionEventRequest,
  VisionEventResponse,
  ActionExecuteResponse,
  EventActionMapping,
} from '../types';

const BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.warn('API Error:', error.message);
    return Promise.reject(error);
  }
);

// ============================================================
// GET /status — Full system + UI + mappings snapshot
// ============================================================

export const fetchStatus = async (): Promise<BackendStatusResponse> => {
  const { data } = await apiClient.get('/status');
  return data;
};

// ============================================================
// GET /vision-event-types — Supported eye event names
// ============================================================

export const fetchVisionEventTypes = async (): Promise<string[]> => {
  const { data } = await apiClient.get('/vision-event-types');
  return data.vision_event_types;
};

// ============================================================
// GET /actions — Supported action names
// ============================================================

export const fetchActions = async (): Promise<string[]> => {
  const { data } = await apiClient.get('/actions');
  return data.actions;
};

// ============================================================
// GET /mappings — Current event→action mappings + supported lists
// ============================================================

export const fetchMappings = async (): Promise<MappingsResponse> => {
  const { data } = await apiClient.get('/mappings');
  return data;
};

// ============================================================
// PUT /mappings — Save edited mappings
// ============================================================

export const updateMappings = async (
  mappings: Partial<EventActionMapping>
): Promise<MappingsResponse> => {
  const { data } = await apiClient.put('/mappings', { mappings });
  return data;
};

// ============================================================
// POST /vision-events — Send an eye/vision event
// ============================================================

export const sendVisionEvent = async (
  payload: VisionEventRequest
): Promise<VisionEventResponse> => {
  const { data } = await apiClient.post('/vision-events', payload);
  return data;
};

// ============================================================
// POST /actions/execute — Manually trigger an action
// ============================================================

export const executeAction = async (
  action: string
): Promise<ActionExecuteResponse> => {
  const { data } = await apiClient.post('/actions/execute', { action });
  return data;
};