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
}));
