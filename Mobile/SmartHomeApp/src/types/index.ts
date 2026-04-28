// ============================================================
// App Mode
// ============================================================

export type AppMode = 'standard' | 'accessibility';

// ============================================================
// Vision / Eye Events
// ============================================================

/** Supported vision event keys (source:name) or string */
export type VisionEventName = string;

/** Maps each vision-event key to an action string. */
export type EventActionMapping = Record<string, string>;

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
    type?: string;
    source: string;
    name: string;
    event_key?: string;
    timestamp: string;
    confidence?: number;
    ignored?: boolean;
    ignore_reason?: string;
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
// Control Mode Types
// ============================================================

export type ControlMode = 'eye_only' | 'hand_only' | 'hybrid';

export interface EnabledSources {
  eye: boolean;
  hand: boolean;
}

export interface ControlModeResponse {
  success: boolean;
  control_mode: ControlMode;
  enabled_sources: EnabledSources;
  supported_control_modes?: string[];
  message?: string;
}

export interface UpdateControlModeRequest {
  control_mode: ControlMode;
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
  control_mode?: ControlMode;
  enabled_sources?: EnabledSources;
}

export interface MappingsResponse {
  success: boolean;
  mappings: EventActionMapping;
  supported_vision_events: string[];
  supported_actions: string[];
}

export interface VisionEventResponse {
  success: boolean;
  ignored?: boolean;
  reason?: string;
  control_mode?: ControlMode;
  enabled_sources?: EnabledSources;
  vision_event: { source: string; name: string; event_key?: string };
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
  ignored?: boolean;
  reason?: string;
  control_mode?: string;
}
