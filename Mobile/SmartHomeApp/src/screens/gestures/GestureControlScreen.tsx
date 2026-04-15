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

  const handleTestEvent = async (name: VisionEventName) => {
    const entryId = `evt-${++historyIdCounter.current}`;
    try {
      const response = await sendVisionEventApi({ source: 'eye', name });
      const entry: EventHistoryEntry = {
        id: entryId,
        timestamp: new Date().toLocaleTimeString(),
        eventName: name,
        mappedAction: response.mapped_action,
        success: response.success,
      };
      setEventHistory(prev => [entry, ...prev].slice(0, 20));
      await fetchBackendStatus();
    } catch {
      const entry: EventHistoryEntry = {
        id: entryId,
        timestamp: new Date().toLocaleTimeString(),
        eventName: name,
        mappedAction: null,
        success: false,
      };
      setEventHistory(prev => [entry, ...prev].slice(0, 20));
      showToast(`Failed to send: ${name}`, 'error');
    }
  };

  const visionEvents: VisionEventName[] = ['look_left', 'look_right', 'short_blink', 'long_blink'];

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Mapping & Diagnostics" />
        <Text style={styles.description}>
          Customize which action each eye event triggers. Changes are saved to the backend.
        </Text>

        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 60 }}>
          {/* ── Event→Action Mappings ── */}
          <Card style={styles.section}>
            <Text style={styles.sectionLabel}>EVENT → ACTION MAPPINGS</Text>
            {visionEvents.map(event => (
              <MappingRow
                key={event}
                eventName={event}
                currentAction={editedMappings[event]}
                availableActions={supportedActions}
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
          <SectionTitle title="Test Events" />
          <View style={styles.testGrid}>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('look_left')}>
              <ArrowLeft color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Left</Text>
            </Pressable>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('look_right')}>
              <ArrowRight color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Right</Text>
            </Pressable>
            <Pressable style={[styles.testButton, styles.testButtonPrimary]} onPress={() => handleTestEvent('short_blink')}>
              <Zap color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Select</Text>
            </Pressable>
            <Pressable style={styles.testButton} onPress={() => handleTestEvent('long_blink')}>
              <RotateCcw color={theme.colors.text.primary} size={20} />
              <Text style={styles.testBtnText}>Back</Text>
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
