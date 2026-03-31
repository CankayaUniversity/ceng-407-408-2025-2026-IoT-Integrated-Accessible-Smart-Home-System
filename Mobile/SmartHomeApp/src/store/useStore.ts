import { create } from 'zustand';
import { AppMode, Device, Climate, SystemMetrics } from '../types';
import { mockDevices, mockClimate, mockSystemMetrics, mockQuickActions, mockGestures, mockRooms, mockSecurity } from '../mocks';

interface AppState {
  appMode: AppMode;
  setAppMode: (mode: AppMode) => void;
  
  devices: Device[];
  toggleDevice: (id: string) => void;
  
  climate: Climate;
  setTargetTemperature: (temp: number) => void;
  
  systemMetrics: SystemMetrics;
  
  quickActions: typeof mockQuickActions;
  gestures: typeof mockGestures;
  rooms: typeof mockRooms;
  security: typeof mockSecurity;
  
  // Feedback States
  isLoading: boolean;
  toastMessage: string | null;
  toastSeverity: 'info' | 'success' | 'warning' | 'error';
  
  toggleDeviceAsync: (id: string) => Promise<void>;
  clearToast: () => void;
  showToast: (msg: string, severity: 'info' | 'success' | 'warning' | 'error') => void;
}

export const useStore = create<AppState>((set) => ({
  appMode: 'standard',
  setAppMode: (mode) => set({ appMode: mode }),
  
  devices: mockDevices,
  toggleDevice: (id) => set((state) => ({
    devices: state.devices.map(d => 
      d.id === id ? { ...d, state: d.state === 'on' ? 'off' : 'on' } : d
    )
  })),
  
  climate: mockClimate,
  setTargetTemperature: (temp) => set((state) => ({
    climate: { ...state.climate, targetTemperature: temp }
  })),
  
  systemMetrics: mockSystemMetrics,
  quickActions: mockQuickActions,
  gestures: mockGestures,
  rooms: mockRooms,
  security: mockSecurity,
  
  isLoading: false,
  toastMessage: null,
  toastSeverity: 'info',
  
  clearToast: () => set({ toastMessage: null }),
  showToast: (msg, severity) => set({ toastMessage: msg, toastSeverity: severity }),
  
  toggleDeviceAsync: async (id: string) => {
    set({ isLoading: true });
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 600));
    
    set((state) => {
      const device = state.devices.find(d => d.id === id);
      if (!device) return { isLoading: false, toastMessage: 'Device not found!', toastSeverity: 'error' };
      
      const newState = (device.state === 'on' ? 'off' : 'on') as 'on' | 'off';
      const updatedDevices = state.devices.map(d => 
        d.id === id ? { ...d, state: newState } : d
      );
      
      return {
        devices: updatedDevices,
        isLoading: false,
        toastMessage: `${device.name} turned ${newState}`,
        toastSeverity: 'success',
        systemMetrics: {
          ...state.systemMetrics,
          lastCommand: `Set ${device.name} to ${newState.toUpperCase()}`
        }
      };
    });
  }
}));
