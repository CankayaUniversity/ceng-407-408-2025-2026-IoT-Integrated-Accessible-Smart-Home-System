import React, { useEffect, useState, useCallback, useRef } from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, Pressable } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { MappingRow } from '../../components/cards/MappingRow';
import { EventHistoryItem } from '../../components/cards/EventHistoryItem';
import { theme } from '../../theme';
import {
  Eye, ArrowLeft, ArrowRight, Zap, RotateCcw,
  Wifi, WifiOff, Monitor, Save, Activity,
  Hand,
} from 'lucide-react-native';
import {
  VisionEventName,
  EventActionMapping,
  EventHistoryEntry,
} from '../../types';
import {
  sendVisionEvent as sendVisionEventApi,
  fetchMappings as fetchMappingsApi,
} from '../../services/apiClient';

export const GestureControlScreen = () => {
  const {
    mappings,
    supportedActions,
    supportedVisionEvents,
    isBackendConnected,
    backendStatus,
    fetchBackendStatus,
    fetchMappingsFromBackend,
    updateMappingsOnBackend,
    showToast,
  } = useStore();

  const [editedMappings, setEditedMappings] = useState<EventActionMapping>({ ...mappings });
  const [isDirty, setIsDirty] = useState(false);
  const [eventHistory, setEventHistory] = useState<EventHistoryEntry[]>([]);
  const historyIdCounter = useRef(0);

  // Load mappings and start polling on mount
  useEffect(() => {
    fetchMappingsFromBackend();
    fetchBackendStatus();
    const interval = setInterval(fetchBackendStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Sync edited mappings when backend mappings change (only if not dirty)
  useEffect(() => {
    if (!isDirty) {
      setEditedMappings({ ...mappings });
    }
  }, [mappings]);

  const handleActionChange = useCallback((event: VisionEventName, action: string) => {
    setEditedMappings(prev => ({ ...prev, [event]: action }));
    setIsDirty(true);
  }, []);

  const handleSave = async () => {
    await updateMappingsOnBackend(editedMappings);
    setIsDirty(false);
  };

  const handleTestEvent = async (source: string, name: string) => {
    const entryId = `evt-${++historyIdCounter.current}`;
    try {
      const response = await sendVisionEventApi(source, name);
      const entry: EventHistoryEntry = {
        id: entryId,
        timestamp: new Date().toLocaleTimeString(),
        eventName: `${source}:${name}`,
        mappedAction: response.mapped_action,
        success: response.success,
        ignored: response.ignored,
        reason: response.reason,
        control_mode: response.control_mode,
      };
      setEventHistory(prev => [entry, ...prev].slice(0, 20));
      // fetchBackendStatus is called inside store action
    } catch {
      const entry: EventHistoryEntry = {
        id: entryId,
        timestamp: new Date().toLocaleTimeString(),
        eventName: `${source}:${name}`,
        mappedAction: null,
        success: false,
      };
      setEventHistory(prev => [entry, ...prev].slice(0, 20));
      showToast(`Failed to send: ${name}`, 'error');
    }
  };

  const defaultEyeEvents = ['eye:look_left', 'eye:look_right', 'eye:short_blink', 'eye:long_blink'];
  const defaultHandEvents = ['hand:swipe_left', 'hand:swipe_right', 'hand:pinch', 'hand:open_palm_hold'];
  
  const eyeEvents = supportedVisionEvents?.filter(e => e.startsWith('eye:'))?.length 
    ? supportedVisionEvents.filter(e => e.startsWith('eye:')) 
    : defaultEyeEvents;
    
  const handEvents = supportedVisionEvents?.filter(e => e.startsWith('hand:'))?.length 
    ? supportedVisionEvents.filter(e => e.startsWith('hand:')) 
    : defaultHandEvents;

  const safeActions = supportedActions?.length > 0 
    ? supportedActions 
    : ['NAV_LEFT', 'NAV_RIGHT', 'SELECT', 'CONFIRM', 'BACK', 'LIGHT_ON', 'LIGHT_OFF', 'PLUG_ON', 'PLUG_OFF'];

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Mapping & Diagnostics" />
        <Text style={styles.description}>
          Customize which action each eye event triggers. Changes are saved to the backend.
        </Text>

        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 60 }}>
          {/* ── Eye Events Mappings ── */}
          <Card style={styles.section}>
            <Text style={styles.sectionLabel}>EYE EVENTS</Text>
            {eyeEvents.map(event => (
              <MappingRow
                key={event}
                eventName={event}
                currentAction={editedMappings[event]}
                availableActions={safeActions}
                onActionChange={(action) => handleActionChange(event, action)}
              />
            ))}
          </Card>

          {/* ── Hand Events Mappings ── */}
          <Card style={styles.section}>
            <Text style={styles.sectionLabel}>HAND EVENTS</Text>
            {handEvents.map(event => (
              <MappingRow
                key={event}
                eventName={event}
                currentAction={editedMappings[event]}
                availableActions={safeActions}
                onActionChange={(action) => handleActionChange(event, action)}
              />
            ))}
          </Card>

          {isDirty && (
            <Button
              title="Save Mappings"
              variant="primary"
              large
              style={styles.saveBtn}
              onPress={handleSave}
            />
          )}

          {/* ── Live Diagnostics ── */}
          <SectionTitle title="Live Diagnostics" />
          <Card style={styles.section}>
            <View style={styles.diagRow}>
              {isBackendConnected ? (
                <Wifi size={16} color={theme.colors.success} />
              ) : (
                <WifiOff size={16} color={theme.colors.error} />
              )}
              <Text style={[styles.diagValue, { color: isBackendConnected ? theme.colors.success : theme.colors.error }]}>
                {isBackendConnected ? 'Connected' : 'Disconnected'}
              </Text>
              {backendStatus?.device_controller_mode && (
                <View style={styles.modeBadge}>
                  <Text style={styles.modeText}>{backendStatus.device_controller_mode}</Text>
                </View>
              )}
            </View>

            <View style={styles.divider} />

            <View style={styles.diagRow}>
              <Eye size={16} color={theme.colors.text.secondary} />
              <Text style={styles.diagLabel}>Last Event</Text>
              <Text style={styles.diagValue}>
                {backendStatus?.system_state?.last_vision_event?.name || '—'}
              </Text>
            </View>

            <View style={styles.diagRow}>
              <Zap size={16} color={theme.colors.text.secondary} />
              <Text style={styles.diagLabel}>Last Action</Text>
              <Text style={styles.diagValue}>
                {backendStatus?.system_state?.last_action?.action || '—'}
              </Text>
            </View>

            <View style={styles.diagRow}>
              <Monitor size={16} color={theme.colors.text.secondary} />
              <Text style={styles.diagLabel}>Screen</Text>
              <Text style={styles.diagValue}>
                {backendStatus?.ui_state?.current_screen || '—'}
              </Text>
            </View>

            <View style={styles.diagRow}>
              <Activity size={16} color={theme.colors.text.secondary} />
              <Text style={styles.diagLabel}>Selected</Text>
              <Text style={styles.diagValue}>
                {backendStatus?.ui_state?.selected_item || '—'}
              </Text>
            </View>
          </Card>

          {/* ── Test Events ── */}
          <SectionTitle title="Test Eye Events" />
          <View style={styles.testGrid}>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('eye', 'look_left')}>
              <ArrowLeft color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Look Left</Text>
            </Pressable>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('eye', 'look_right')}>
              <ArrowRight color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Look Right</Text>
            </Pressable>
            <Pressable style={[styles.testButton, styles.testButtonPrimary]} onPress={() => handleTestEvent('eye', 'short_blink')}>
              <Zap color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Short Blink</Text>
            </Pressable>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('eye', 'long_blink')}>
              <RotateCcw color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Long Blink</Text>
            </Pressable>
          </View>

          <SectionTitle title="Test Hand Events" />
          <View style={styles.testGrid}>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('hand', 'swipe_left')}>
              <ArrowLeft color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Swipe Left</Text>
            </Pressable>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('hand', 'swipe_right')}>
              <ArrowRight color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Swipe Right</Text>
            </Pressable>
            <Pressable style={[styles.testButton, styles.testButtonPrimary]} onPress={() => handleTestEvent('hand', 'pinch')}>
              <Hand color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Pinch</Text>
            </Pressable>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('hand', 'open_palm_hold')}>
              <Hand color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Palm Hold</Text>
            </Pressable>
          </View>

          {/* ── Event History ── */}
          {eventHistory.length > 0 && (
            <>
              <SectionTitle title={`Event History (${eventHistory.length})`} />
              <Card style={styles.section}>
                {eventHistory.map(entry => (
                  <EventHistoryItem key={entry.id} entry={entry} />
                ))}
              </Card>
            </>
          )}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  container: {
    flex: 1,
    paddingHorizontal: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
  },
  description: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.lg,
  },
  section: {
    marginBottom: theme.spacing.md,
  },
  sectionLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
    letterSpacing: 1.5,
    marginBottom: theme.spacing.md,
  },
  saveBtn: {
    marginBottom: theme.spacing.lg,
  },
  // Diagnostics
  diagRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    paddingVertical: theme.spacing.xs,
  },
  diagLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    flex: 1,
  },
  diagValue: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
  },
  modeBadge: {
    backgroundColor: theme.colors.surfaceHighlight,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.radius.sm,
    marginLeft: 'auto',
  },
  modeText: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
    fontSize: 10,
  },
  divider: {
    height: 1,
    backgroundColor: theme.colors.surfaceHighlight,
    marginVertical: theme.spacing.sm,
  },
  // Test buttons
  testGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.md,
    justifyContent: 'space-between',
    marginBottom: theme.spacing.lg,
  },
  testButton: {
    width: '47%',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: theme.spacing.lg,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.lg,
    gap: theme.spacing.sm,
  },
  testButtonPrimary: {
    backgroundColor: theme.colors.primaryDark,
  },
  testBtnText: {
    ...theme.typography.bodyLarge,
    fontWeight: 'bold',
    color: theme.colors.text.primary,
  },
});
