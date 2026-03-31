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
  apiStatus: 'online' | 'offline' | 'connecting';
  lastCommand: string;
  lastEvent: string;
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

// Backend Navigation Interfaces
export interface BackendUIState {
  current_screen: string;
  selected_index: number;
  selected_item: string;
  items: string[];
}

export interface BackendSystemState {
  last_event: { event_type: string; name: string; timestamp: string } | null;
  last_intent: { intent: string; timestamp: string } | null;
  last_command: { command: string; timestamp: string } | null;
  device_status: { light: string; plug: string };
}

export interface BackendStatusResponse {
  success: boolean;
  system_state: BackendSystemState;
  ui_state: BackendUIState;
}
