import React, { useEffect, useState } from 'react';
import { View, StyleSheet, Text, ActivityIndicator, Pressable, ScrollView } from 'react-native';
import { useStore } from '../../store/useStore';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { Eye, Power, ArrowLeft, ArrowRight, Zap, RotateCcw } from 'lucide-react-native';
import { VisionEventName } from '../../types';

export const DashboardAccessible = () => {
  const {
    showToast,
    backendStatus,
    isBackendConnected,
    fetchBackendStatus,
    sendVisionEvent,
    mappings,
  } = useStore();
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  useEffect(() => {
    const load = async () => {
      await fetchBackendStatus();
      setIsInitialLoad(false);
    };
    load();
    const interval = setInterval(fetchBackendStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleEyeEvent = async (name: VisionEventName) => {
    try {
      await sendVisionEvent(name);
    } catch {
      showToast(`Failed to send event: ${name}`, 'error');
    }
  };

  if (isInitialLoad && !backendStatus) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Connecting to Backend...</Text>
      </View>
    );
  }

  if (!backendStatus) {
    return (
      <View style={[styles.container, styles.centered]}>
        <Text style={styles.errorText}>No connection to API</Text>
        <Pressable style={styles.retryBtn} onPress={fetchBackendStatus}>
          <RotateCcw size={20} color={theme.colors.text.primary} />
          <Text style={styles.retryText}>Retry</Text>
        </Pressable>
      </View>
    );
  }

  const { ui_state, system_state } = backendStatus;

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 100 }}>
      {/* Title Header from backend current_screen */}
      <View style={styles.header}>
        <Eye size={28} color={theme.colors.primary} />
        <Text style={styles.title}>{ui_state.current_screen.replace(/_/g, ' ').toUpperCase()}</Text>
      </View>

      {/* Backend-Driven Menu Items — Horizontal for gaze-friendliness */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.menuScroll}
        style={styles.menuScrollOuter}
      >
        {ui_state.items.map((item) => {
          const isSelected = ui_state.selected_item === item;
          return (
            <Card
              key={item}
              style={[styles.menuItem, isSelected && styles.menuItemSelected]}
              focused={isSelected}
            >
              <Text style={[styles.itemText, isSelected && styles.itemTextSelected]}>
                {item}
              </Text>
              {isSelected && <Zap size={20} color={theme.colors.primary} />}
            </Card>
          );
        })}
      </ScrollView>

      {/* Active Mapping Display */}
      <View style={styles.mappingSection}>
        <Text style={styles.sectionTitle}>Active Mappings</Text>
        <View style={styles.mappingGrid}>
          {(Object.entries(mappings) as [VisionEventName, string][]).map(([event, action]) => (
            <View key={event} style={styles.mappingChip}>
              <Text style={styles.mappingEvent}>{event.replace(/_/g, ' ')}</Text>
              <Text style={styles.mappingArrow}>→</Text>
              <Text style={styles.mappingAction}>{action}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Device Status */}
      <View style={styles.statusSection}>
        <Text style={styles.sectionTitle}>Physical Devices</Text>
        <View style={styles.deviceRow}>
          {Object.entries(system_state.device_status).map(([name, status]) => (
            <Card key={name} style={styles.deviceIndicator}>
              <Text style={styles.deviceLabel}>{name.charAt(0).toUpperCase() + name.slice(1)}</Text>
              <Power
                size={24}
                color={status === 'on' ? theme.colors.success : theme.colors.text.secondary}
              />
            </Card>
          ))}
        </View>
      </View>

      {/* Test Buttons */}
      <View style={styles.testSection}>
        <Text style={styles.sectionTitle}>API Testing Controls</Text>
        <View style={styles.testGrid}>
          <Pressable style={styles.testButton} onPress={() => handleEyeEvent('look_left')}>
            <ArrowLeft color={theme.colors.text.primary} />
            <Text style={styles.testBtnText}>Left</Text>
          </Pressable>
          <Pressable style={styles.testButton} onPress={() => handleEyeEvent('look_right')}>
            <ArrowRight color={theme.colors.text.primary} />
            <Text style={styles.testBtnText}>Right</Text>
          </Pressable>
          <Pressable style={[styles.testButton, { backgroundColor: theme.colors.primaryDark }]} onPress={() => handleEyeEvent('short_blink')}>
            <Zap color={theme.colors.text.primary} />
            <Text style={styles.testBtnText}>Select</Text>
          </Pressable>
          <Pressable style={styles.testButton} onPress={() => handleEyeEvent('long_blink')}>
            <RotateCcw color={theme.colors.text.primary} />
            <Text style={styles.testBtnText}>Back</Text>
          </Pressable>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingVertical: theme.spacing.md,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    ...theme.typography.subtitle,
    color: theme.colors.text.secondary,
    marginTop: theme.spacing.md,
  },
  errorText: {
    ...theme.typography.h2,
    color: theme.colors.error,
  },
  retryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.lg,
    padding: theme.spacing.md,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.lg,
  },
  retryText: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.xl,
    paddingHorizontal: theme.spacing.sm,
  },
  title: {
    ...theme.typography.h1,
    color: theme.colors.text.primary,
    marginLeft: theme.spacing.md,
  },
  // Horizontal menu
  menuScrollOuter: {
    marginBottom: theme.spacing.xl,
  },
  menuScroll: {
    paddingHorizontal: theme.spacing.xs,
    gap: theme.spacing.md,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: theme.spacing.xxl,
    paddingVertical: theme.spacing.xl,
    backgroundColor: '#1E2333',
    borderWidth: 2,
    borderColor: 'transparent',
    minWidth: 140,
    minHeight: theme.accessibility.cardMinHeight,
    gap: theme.spacing.sm,
  },
  menuItemSelected: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.focusBackground,
  },
  itemText: {
    fontSize: 22,
    fontWeight: '600',
    color: theme.colors.text.secondary,
  },
  itemTextSelected: {
    color: theme.colors.primary,
    fontSize: 26,
    fontWeight: '700',
  },
  // Mapping display
  mappingSection: {
    marginBottom: theme.spacing.xl,
  },
  sectionTitle: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.md,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  mappingGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
  },
  mappingChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.xs,
    backgroundColor: theme.colors.surfaceHighlight,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.radius.lg,
  },
  mappingEvent: {
    ...theme.typography.caption,
    color: theme.colors.text.primary,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  mappingArrow: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
  },
  mappingAction: {
    ...theme.typography.caption,
    color: theme.colors.primaryLight,
    fontWeight: '600',
  },
  // Devices
  statusSection: {
    marginBottom: theme.spacing.xxl,
  },
  deviceRow: {
    flexDirection: 'row',
    gap: theme.spacing.md,
  },
  deviceIndicator: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: theme.spacing.lg,
  },
  deviceLabel: {
    ...theme.typography.h3,
    color: theme.colors.text.primary,
  },
  // Test buttons
  testSection: {
    marginTop: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
    borderTopWidth: 1,
    borderTopColor: theme.colors.border,
  },
  testGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.md,
    justifyContent: 'space-between',
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
  testBtnText: {
    ...theme.typography.bodyLarge,
    fontWeight: 'bold',
    color: theme.colors.text.primary,
  }
});
