// ============================================================
// App Mode
// ============================================================

export type AppMode = 'standard' | 'accessibility';

// ============================================================
// Vision / Eye Events
// ============================================================

/** The only four supported eye-event names (no look_up / look_down). */
export type VisionEventName = 'look_left' | 'look_right' | 'short_blink' | 'long_blink';

/** Maps each vision-event name to an action string. */
export type EventActionMapping = Record<VisionEventName, string>;

/** POST /vision-events request body. */
export interface VisionEventRequest {
  source: string;
  name: VisionEventName;
  confidence?: number;
  timestamp?: number | string;
  metadata?: Record<string, any>;
}

// ============================================================
// Backend State Models (GET /status)
// ============================================================

export interface BackendUIState {
  current_screen: string;
  selected_index: number;
  selected_item: string;
  items: string[];
}

export interface BackendSystemState {
  last_vision_event: {
    type: string;
    source: string;
    name: string;
    timestamp: string;
    confidence?: number;
  } | null;
  last_action: {
    action: string;
    timestamp: string;
    trigger: string;
    trigger_detail?: string;
  } | null;
  last_command: {
    command: string;
    timestamp: string;
    hardware_result?: any;
  } | null;
  device_status: Record<string, string>;
}

// ============================================================
// Backend API Responses
// ============================================================

export interface BackendStatusResponse {
  success: boolean;
  system_state: BackendSystemState;
  ui_state: BackendUIState;
  device_controller_mode: string;
  mappings: EventActionMapping;
}

export interface MappingsResponse {
  success: boolean;
  mappings: EventActionMapping;
  supported_vision_events: string[];
  supported_actions: string[];
}

export interface VisionEventResponse {
  success: boolean;
  vision_event: { source: string; name: string };
  mapped_action: string | null;
  action_result?: any;
  ui_state: BackendUIState;
  device_status: Record<string, string>;
}

export interface ActionExecuteResponse {
  success: boolean;
  executed_action: string;
  result: any;
  ui_state: BackendUIState;
  device_status: Record<string, string>;
}

// ============================================================
// Device Models
// ============================================================

export type DeviceType = 'light' | 'thermostat' | 'lock' | 'sensor' | 'camera' | 'speaker';
export type DeviceState = 'on' | 'off' | 'offline' | 'warning';

export interface Device {
  id: string;
  name: string;
  type: DeviceType;
  roomId: string;
  state: DeviceState;
  value?: number | string;
}

// ============================================================
// Room
// ============================================================

export interface Room {
  id: string;
  name: string;
  category: 'indoor' | 'outdoor';
  devicesCount: number;
  imageUrl?: string;
}

// ============================================================
// Climate
// ============================================================

export interface Climate {
  temperature: number;
  humidity: number;
  airQuality: number;
  mode: 'cool' | 'heat' | 'auto';
  targetTemperature: number;
  rooms: { name: string; temp: number }[];
}

// ============================================================
// System Metrics (local mock for standard mode display)
// ============================================================

export interface SystemMetrics {
  cpuLoad: number;
  memoryUsage: number;
  uptime: string;
  apiStatus: 'online' | 'offline' | 'connecting';
  lastCommand: string;
  lastEvent: string;
  neuralEngine: string;
}

// ============================================================
// Quick Actions
// ============================================================

export interface QuickAction {
  id: string;
  label: string;
  iconName: string;
  action: string;
}

// ============================================================
// Security
// ============================================================

export interface SecurityStatus {
  armed: boolean;
  doorsLocked: boolean;
  windowsClosed: boolean;
  activeAlarms: number;
}

// ============================================================
// Event History (local UI state for diagnostics)
// ============================================================

export interface EventHistoryEntry {
  id: string;
  timestamp: string;
  eventName: string;
  mappedAction: string | null;
  success: boolean;
}
