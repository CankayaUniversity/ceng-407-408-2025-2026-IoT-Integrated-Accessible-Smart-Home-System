import { create } from 'zustand';
import {
  AppMode,
  Device,
  Climate,
  SystemMetrics,
  BackendStatusResponse,
  EventActionMapping,
  VisionEventName,
  ControlMode,
  EnabledSources,
  VisionEventResponse,
} from '../types';
import {
  mockDevices,
  mockClimate,
  mockSystemMetrics,
  mockQuickActions,
  mockRooms,
  mockSecurity,
  mockMappings,
  mockSupportedActions,
  mockSupportedVisionEvents,
} from '../mocks';
import {
  fetchStatus,
  fetchMappings as fetchMappingsApi,
  updateMappings as updateMappingsApi,
  sendVisionEvent as sendVisionEventApi,
  executeAction as executeActionApi,
  fetchControlMode as fetchControlModeApi,
  updateControlMode as updateControlModeApi,
} from '../services/apiClient';
import { speakFeedback } from '../services/feedbackService';

interface AppState {
  // ── App Mode ──
  appMode: AppMode;
  setAppMode: (mode: AppMode) => void;

  // ── Local Device State (mock / standard mode) ──
  devices: Device[];
  toggleDevice: (id: string) => void;

  // ── Climate ──
  climate: Climate;
  setTargetTemperature: (temp: number) => void;

  // ── System Metrics (local mock) ──
  systemMetrics: SystemMetrics;

  // ── Standard-mode mock data ──
  quickActions: typeof mockQuickActions;
  rooms: typeof mockRooms;
  security: typeof mockSecurity;

  // ── Backend-driven State ──
  backendStatus: BackendStatusResponse | null;
  mappings: EventActionMapping;
  supportedActions: string[];
  supportedVisionEvents: string[];
  isBackendConnected: boolean;
  controlMode: ControlMode;
  enabledSources: EnabledSources;
  supportedControlModes: string[];

  // ── Feedback States ──
  isLoading: boolean;
  toastMessage: string | null;
  toastSeverity: 'info' | 'success' | 'warning' | 'error';
  isVoiceFeedbackEnabled: boolean;
  setVoiceFeedbackEnabled: (enabled: boolean) => void;
  lastSpokenActionTimestamp: string | null;
  // ── User
  userName: string;
  setUserName: (name: string) => void;

  // ── Actions ──
  toggleDeviceAsync: (id: string) => Promise<void>;
  clearToast: () => void;
  showToast: (msg: string, severity: 'info' | 'success' | 'warning' | 'error') => void;

  // ── Backend Actions ──
  fetchBackendStatus: () => Promise<void>;
  fetchMappingsFromBackend: () => Promise<void>;
  updateMappingsOnBackend: (mappings: Partial<EventActionMapping>) => Promise<void>;
  sendVisionEvent: (source: string, name: string) => Promise<VisionEventResponse>;
  executeActionOnBackend: (action: string) => Promise<void>;
  fetchControlModeFromBackend: () => Promise<void>;
  updateControlModeOnBackend: (mode: ControlMode) => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  // ── App Mode ──
  appMode: 'standard',
  setAppMode: (mode) => set({ appMode: mode }),

  // ── Devices ──
  devices: mockDevices,
  toggleDevice: (id) => set((state) => ({
    devices: state.devices.map(d =>
      d.id === id ? { ...d, state: d.state === 'on' ? 'off' : 'on' } : d
    )
  })),

  // ── Climate ──
  climate: mockClimate,
  setTargetTemperature: (temp) => set((state) => ({
    climate: { ...state.climate, targetTemperature: temp }
  })),

  // ── System Metrics ──
  systemMetrics: mockSystemMetrics,

  // ── Standard-mode mock data ──
  quickActions: mockQuickActions,
  rooms: mockRooms,
  security: mockSecurity,

  // ── Backend-driven State ──
  backendStatus: null,
  mappings: mockMappings,
  supportedActions: mockSupportedActions,
  supportedVisionEvents: mockSupportedVisionEvents,
  isBackendConnected: false,
  controlMode: 'eye_only',
  enabledSources: { eye: true, hand: false },
  supportedControlModes: ['eye_only', 'hand_only'],

  // ── Feedback ──
  isLoading: false,
  toastMessage: null,
  toastSeverity: 'info',
  isVoiceFeedbackEnabled: true,
  setVoiceFeedbackEnabled: (enabled) => set({ isVoiceFeedbackEnabled: enabled }),
  lastSpokenActionTimestamp: null,
  // User
  userName: 'Kullanıcı',
  setUserName: (name: string) => set({ userName: name }),
  clearToast: () => set({ toastMessage: null }),
  showToast: (msg, severity) => set({ toastMessage: msg, toastSeverity: severity }),

  // ── Toggle Device (async with toast) ──
  toggleDeviceAsync: async (id: string) => {
    set({ isLoading: true });
    await new Promise(resolve => setTimeout(resolve, 600));

    set((state) => {
      const device = state.devices.find(d => d.id === id);
      if (!device) return { isLoading: false, toastMessage: 'Device not found!', toastSeverity: 'error' as const };

      const newState = (device.state === 'on' ? 'off' : 'on') as 'on' | 'off';
      const updatedDevices = state.devices.map(d =>
        d.id === id ? { ...d, state: newState } : d
      );

      return {
        devices: updatedDevices,
        isLoading: false,
        toastMessage: `${device.name} turned ${newState}`,
        toastSeverity: 'success' as const,
        systemMetrics: {
          ...state.systemMetrics,
          lastCommand: `Set ${device.name} to ${newState.toUpperCase()}`
        }
      };
    });
  },

  // ── Backend: Fetch Status ──
  fetchBackendStatus: async () => {
    try {
      const data = await fetchStatus();
      const lastAction = data.system_state.last_action;
      const prevTimestamp = get().lastSpokenActionTimestamp;

      if (
        lastAction?.feedback?.speak &&
        lastAction.feedback.message &&
        lastAction.timestamp !== prevTimestamp
      ) {
        speakFeedback(lastAction.feedback, get().isVoiceFeedbackEnabled);
        set({ lastSpokenActionTimestamp: lastAction.timestamp });
      }

      set({
        backendStatus: data,
        isBackendConnected: true,
        mappings: data.mappings,
        controlMode: data.control_mode || get().controlMode,
        enabledSources: data.enabled_sources || get().enabledSources,
        systemMetrics: {
          ...get().systemMetrics,
          apiStatus: 'online',
          lastEvent: data.system_state.last_vision_event?.name
            ? `${data.system_state.last_vision_event.source}: ${data.system_state.last_vision_event.name}`
            : get().systemMetrics.lastEvent,
          lastCommand: data.system_state.last_command?.command || get().systemMetrics.lastCommand,
        },
      });
    } catch {
      set({ isBackendConnected: false });
    }
  },

  // ── Backend: Fetch Mappings ──
  fetchMappingsFromBackend: async () => {
    try {
      const data = await fetchMappingsApi();
      set({
        mappings: data.mappings,
        supportedActions: data.supported_actions,
        supportedVisionEvents: data.supported_vision_events,
      });
    } catch {
      get().showToast('Failed to load mappings', 'error');
    }
  },

  // ── Backend: Update Mappings ──
  updateMappingsOnBackend: async (newMappings) => {
    try {
      const data = await updateMappingsApi(newMappings);
      set({ mappings: data.mappings });
      get().showToast('Mappings saved', 'success');
    } catch {
      get().showToast('Failed to save mappings', 'error');
    }
  },

  // ── Backend: Send Vision Event ──
  sendVisionEvent: async (source: string, name: string) => {
    try {
      const response = await sendVisionEventApi({ source, name: name as VisionEventName });
      // Refresh status after event
      await get().fetchBackendStatus();
      if (response.feedback) {
        speakFeedback(response.feedback, get().isVoiceFeedbackEnabled);
      }
      return response;
    } catch (err: any) {
      get().showToast(`Failed to send event: ${name}`, 'error');
      
      const errData = err?.response?.data?.detail;
      if (errData && typeof errData === 'object' && errData.feedback) {
        speakFeedback(errData.feedback, get().isVoiceFeedbackEnabled);
      } else {
        speakFeedback({ message: 'Backend connection failed', severity: 'error', speak: true }, get().isVoiceFeedbackEnabled);
      }
      throw err;
    }
  },

  // ── Backend: Execute Action ──
  executeActionOnBackend: async (action: string) => {
    try {
      const response = await executeActionApi(action);
      await get().fetchBackendStatus();
      if (response.feedback) {
        speakFeedback(response.feedback, get().isVoiceFeedbackEnabled);
      }
    } catch (err: any) {
      get().showToast(`Failed to execute: ${action}`, 'error');
      
      const errData = err?.response?.data?.detail;
      if (errData && typeof errData === 'object' && errData.feedback) {
        speakFeedback(errData.feedback, get().isVoiceFeedbackEnabled);
      } else {
        speakFeedback({ message: 'Command failed', severity: 'error', speak: true }, get().isVoiceFeedbackEnabled);
      }
    }
  },

  // ── Backend: Fetch Control Mode ──
  fetchControlModeFromBackend: async () => {
    try {
      const data = await fetchControlModeApi();
      set({
        controlMode: data.control_mode,
        enabledSources: data.enabled_sources,
        supportedControlModes: data.supported_control_modes || ['eye_only', 'hand_only'],
      });
    } catch {
      // Silently fail or use status fallback
    }
  },

  // ── Backend: Update Control Mode ──
  updateControlModeOnBackend: async (mode: ControlMode) => {
    try {
      const data = await updateControlModeApi({ control_mode: mode });
      set({
        controlMode: data.control_mode,
        enabledSources: data.enabled_sources,
      });
      get().showToast('Control mode updated', 'success');
      await get().fetchBackendStatus();
    } catch {
      get().showToast('Failed to update control mode', 'error');
    }
  },
}));
