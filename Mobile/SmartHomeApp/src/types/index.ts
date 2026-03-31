export type AppMode = 'standard' | 'accessibility';

export type DeviceType = 'light' | 'thermostat' | 'lock' | 'sensor' | 'camera' | 'speaker';
export type DeviceState = 'on' | 'off' | 'offline' | 'warning';

export interface Device {
  id: string;
  name: string;
  type: DeviceType;
  roomId: string;
  state: DeviceState;
  value?: number | string; // e.g. brightness 0-100 or temp 22
}

export interface Room {
  id: string;
  name: string;
  category: 'indoor' | 'outdoor';
  devicesCount: number;
  imageUrl?: string;
  // simplified for mockup
}

export interface Climate {
  temperature: number;
  humidity: number;
  airQuality: number; // 0-100
  mode: 'cool' | 'heat' | 'auto';
  targetTemperature: number;
  rooms: { name: string; temp: number }[];
}

export interface SystemMetrics {
  cpuLoad: number;
  memoryUsage: number;
  uptime: string;
  linkStatus: 'connected' | 'error' | 'connecting';
  lastCommand: string;
  lastGesture: string;
  neuralEngine: string;
}

export interface Gesture {
  id: string;
  name: string;
  description: string;
  iconName: string;
  assignedAction: string;
}

export interface QuickAction {
  id: string;
  label: string;
  iconName: string;
  action: string;
}

export interface SecurityStatus {
  armed: boolean;
  doorsLocked: boolean;
  windowsClosed: boolean;
  activeAlarms: number;
}
