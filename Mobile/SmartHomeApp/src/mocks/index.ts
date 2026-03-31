import { Device, Room, Climate, SystemMetrics, Gesture, QuickAction, SecurityStatus } from '../types';

export const mockDevices: Device[] = [
  { id: '1', name: 'Living Room Lights', type: 'light', roomId: 'r1', state: 'on', value: 80 },
  { id: '2', name: 'Main Thermostat', type: 'thermostat', roomId: 'r1', state: 'on', value: 22 },
  { id: '3', name: 'Front Door Lock', type: 'lock', roomId: 'r2', state: 'on' },
  { id: '4', name: 'Kitchen Camera', type: 'camera', roomId: 'r3', state: 'offline' },
  { id: '5', name: 'Bedroom AC', type: 'thermostat', roomId: 'r4', state: 'off', value: 18 },
  { id: '6', name: 'Patio Sensor', type: 'sensor', roomId: 'r5', state: 'warning' },
];

export const mockRooms: Room[] = [
  { id: 'r1', name: 'Living Room', category: 'indoor', devicesCount: 4 },
  { id: 'r2', name: 'Entrance', category: 'indoor', devicesCount: 2 },
  { id: 'r3', name: 'Kitchen', category: 'indoor', devicesCount: 5 },
  { id: 'r4', name: 'Bedroom', category: 'indoor', devicesCount: 3 },
  { id: 'r5', name: 'Patio', category: 'outdoor', devicesCount: 2 },
];

export const mockClimate: Climate = {
  temperature: 22.5,
  humidity: 45,
  airQuality: 92,
  mode: 'auto',
  targetTemperature: 22,
  rooms: [
    { name: 'Living Room', temp: 22 },
    { name: 'Bedroom', temp: 19 },
    { name: 'Kitchen', temp: 24 },
  ]
};

export const mockSystemMetrics: SystemMetrics = {
  cpuLoad: 24.5,
  memoryUsage: 64.2,
  uptime: '15d 14h 22m',
  apiStatus: 'online',
  lastCommand: 'Set Living Room Light 80%',
  lastEvent: 'Gaze Detected: Look Right',
  neuralEngine: 'Active (Node A2)',
};

export const mockGestures: Gesture[] = [
  { id: 'g1', name: 'Swipe Up', description: 'Raise hand upwards', iconName: 'arrow-up', assignedAction: 'Turn All Lights On' },
  { id: 'g2', name: 'Swipe Down', description: 'Lower hand downwards', iconName: 'arrow-down', assignedAction: 'Turn All Lights Off' },
  { id: 'g3', name: 'Circle', description: 'Draw a circle in air', iconName: 'refresh-ccw', assignedAction: 'Toggle AC' },
];

export const mockQuickActions: QuickAction[] = [
  { id: 'qa1', label: 'Good Morning', iconName: 'sun', action: 'morning_routine' },
  { id: 'qa2', label: 'Leaving Home', iconName: 'log-out', action: 'leave_routine' },
  { id: 'qa3', label: 'Sleep', iconName: 'moon', action: 'sleep_routine' },
  { id: 'qa4', label: 'Movie Mode', iconName: 'film', action: 'movie_routine' },
];

export const mockSecurity: SecurityStatus = {
  armed: true,
  doorsLocked: true,
  windowsClosed: false,
  activeAlarms: 0,
};
