import React, { useEffect, useState } from 'react';
import { View, StyleSheet, Text, ActivityIndicator, Pressable, ScrollView } from 'react-native';
import { useStore } from '../../store/useStore';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { Eye, Power, ArrowLeft, ArrowRight, Zap, RefreshCw } from 'lucide-react-native';
import { fetchStatus, sendEyeEvent } from '../../services/apiClient';
import { BackendStatusResponse } from '../../types';

export const DashboardAccessible = () => {
  const { showToast } = useStore();
  const [status, setStatus] = useState<BackendStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadStatus = async () => {
    try {
      const data = await fetchStatus();
      setStatus(data);
    } catch (error) {
      showToast('API Connection Failed', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
    // Poll every 2 seconds to keep in sync with API if external events occur
    const interval = setInterval(loadStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleEyeEvent = async (name: 'look_left' | 'look_right' | 'short_blink' | 'long_blink') => {
    try {
      await sendEyeEvent(name);
      await loadStatus(); // Refresh immediately after interaction
    } catch (error) {
      showToast(`Failed to send event: ${name}`, 'error');
    }
  };

  if (isLoading && !status) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Connecting to Eye Node...</Text>
      </View>
    );
  }

  if (!status) {
    return (
      <View style={[styles.container, styles.centered]}>
        <Text style={styles.errorText}>No connection to API</Text>
      </View>
    );
  }

  const { ui_state, system_state } = status;

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 100 }}>
      {/* Title Header directly driven from Backend current_screen */}
      <View style={styles.header}>
        <Eye size={28} color={theme.colors.primary} />
        <Text style={styles.title}>{ui_state.current_screen.replace('_', ' ').toUpperCase()}</Text>
      </View>

      {/* Backend Driven Menu Items */}
      <View style={styles.menuContainer}>
        {ui_state.items.map((item) => {
          const isSelected = ui_state.selected_item === item;
          return (
            <Card 
              key={item} 
              style={[
                styles.menuItem, 
                isSelected && styles.menuItemSelected
              ]}
            >
              <Text style={[styles.itemText, isSelected && styles.itemTextSelected]}>
                {item}
              </Text>
              {isSelected && <Zap size={24} color={theme.colors.primary} />}
            </Card>
          );
        })}
      </View>

      {/* Device Status Information */}
      <View style={styles.statusSection}>
        <Text style={styles.sectionTitle}>Physical Devices</Text>
        <View style={styles.deviceRow}>
          <Card style={styles.deviceIndicator}>
            <Text style={styles.deviceLabel}>Light</Text>
            <Power 
              size={24} 
              color={system_state.device_status.light === 'on' ? theme.colors.success : theme.colors.text.secondary} 
            />
          </Card>
          <Card style={styles.deviceIndicator}>
            <Text style={styles.deviceLabel}>Plug</Text>
            <Power 
              size={24} 
              color={system_state.device_status.plug === 'on' ? theme.colors.success : theme.colors.text.secondary} 
            />
          </Card>
        </View>
      </View>

      {/* Test Buttons Section for Simulate API Events */}
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
            <RefreshCw color={theme.colors.text.primary} />
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
  menuContainer: {
    gap: theme.spacing.md,
    marginBottom: theme.spacing.xxl,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: theme.spacing.xl,
    backgroundColor: '#1E2333',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  menuItemSelected: {
    borderColor: theme.colors.primary,
    backgroundColor: 'rgba(232, 117, 88, 0.1)',
  },
  itemText: {
    fontSize: 24,
    fontWeight: '600',
    color: theme.colors.text.secondary,
  },
  itemTextSelected: {
    color: theme.colors.primary,
    fontSize: 28,
  },
  statusSection: {
    marginBottom: theme.spacing.xxl,
  },
  sectionTitle: {
    ...theme.typography.subtitle,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.md,
    textTransform: 'uppercase',
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
