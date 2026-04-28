import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, Pressable } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { Activity, Cpu, HardDrive, Network, Zap, Eye, Monitor, Power, RefreshCw } from 'lucide-react-native';
import { VisionEventName } from '../../types';

const MetricRow = ({ icon, label, value, statusColor }: any) => (
  <View style={styles.metricRow}>
    <View style={styles.metricLeft}>
      {icon}
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
    <Text style={[styles.metricValue, { color: statusColor || theme.colors.text.primary }]}>
      {value}
    </Text>
  </View>
);

export const SystemStatusScreen = () => {
  const { systemMetrics, backendStatus, isBackendConnected, fetchBackendStatus, mappings, controlMode, enabledSources } = useStore();

  useEffect(() => {
    fetchBackendStatus();
    const interval = setInterval(fetchBackendStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const systemState = backendStatus?.system_state;

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <View style={styles.titleRow}>
          <SectionTitle title="System Status" />
          <Pressable style={styles.refreshBtn} onPress={fetchBackendStatus}>
            <RefreshCw size={18} color={theme.colors.text.secondary} />
          </Pressable>
        </View>
        
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Connection & API Status */}
          <Card style={styles.card}>
            <MetricRow 
              icon={<Network size={20} color={theme.colors.text.secondary} />} 
              label="API Status" 
              value={isBackendConnected ? 'CONNECTED' : 'DISCONNECTED'} 
              statusColor={isBackendConnected ? theme.colors.success : theme.colors.error} 
            />
            <View style={styles.divider} />
            <MetricRow 
              icon={<Monitor size={20} color={theme.colors.text.secondary} />} 
              label="Controller Mode" 
              value={backendStatus?.device_controller_mode?.toUpperCase() || 'UNKNOWN'} 
              statusColor={theme.colors.text.secondary}
            />
            <View style={styles.divider} />
            <MetricRow 
              icon={<Activity size={20} color={theme.colors.text.secondary} />} 
              label="Control Mode" 
              value={controlMode.toUpperCase()} 
              statusColor={theme.colors.primary}
            />
            <View style={styles.divider} />
            <MetricRow 
              icon={<Eye size={20} color={theme.colors.text.secondary} />} 
              label="Enabled Sources" 
              value={`Eye: ${enabledSources?.eye ? 'ON' : 'OFF'} | Hand: ${enabledSources?.hand ? 'ON' : 'OFF'}`} 
            />
          </Card>

          {/* Last Vision Event */}
          <SectionTitle title="Last Vision Event" />
          <Card style={styles.card}>
            {systemState?.last_vision_event ? (
              <>
                <MetricRow 
                  icon={<Eye size={20} color={theme.colors.primary} />} 
                  label="Event" 
                  value={systemState.last_vision_event.event_key || systemState.last_vision_event.name} 
                />
                <View style={styles.divider} />
                <MetricRow 
                  icon={<Activity size={20} color={theme.colors.text.secondary} />} 
                  label="Source" 
                  value={systemState.last_vision_event.source} 
                />
                {systemState.last_vision_event.ignored && (
                  <>
                    <View style={styles.divider} />
                    <MetricRow 
                      icon={<Activity size={20} color={theme.colors.warning} />} 
                      label="Status" 
                      value={`IGNORED: ${systemState.last_vision_event.ignore_reason || 'Source disabled'}`} 
                      statusColor={theme.colors.warning}
                    />
                  </>
                )}
                <View style={styles.divider} />
                <Text style={styles.timestampText}>
                  {new Date(systemState.last_vision_event.timestamp).toLocaleTimeString()}
                </Text>
              </>
            ) : (
              <Text style={styles.noDataText}>No vision events yet</Text>
            )}
          </Card>

          {/* Last Action */}
          <SectionTitle title="Last Action" />
          <Card style={styles.card}>
            {systemState?.last_action ? (
              <>
                <MetricRow 
                  icon={<Zap size={20} color={theme.colors.warning} />} 
                  label="Action" 
                  value={systemState.last_action.action} 
                />
                <View style={styles.divider} />
                <MetricRow 
                  icon={<Activity size={20} color={theme.colors.text.secondary} />} 
                  label="Trigger" 
                  value={`${systemState.last_action.trigger}${systemState.last_action.trigger_detail ? ` (${systemState.last_action.trigger_detail})` : ''}`} 
                />
              </>
            ) : (
              <Text style={styles.noDataText}>No actions yet</Text>
            )}
          </Card>

          {/* Last Command */}
          <SectionTitle title="Last Command" />
          <Card style={styles.card}>
            {systemState?.last_command ? (
              <MetricRow 
                icon={<Power size={20} color={theme.colors.success} />} 
                label="Command" 
                value={systemState.last_command.command} 
              />
            ) : (
              <Text style={styles.noDataText}>No device commands yet</Text>
            )}
          </Card>

          {/* Device Status */}
          <SectionTitle title="Device Status" />
          <Card style={styles.card}>
            {systemState?.device_status ? (
              Object.entries(systemState.device_status).map(([name, status], idx, arr) => (
                <View key={name}>
                  <MetricRow
                    icon={<Power size={20} color={status === 'on' ? theme.colors.success : theme.colors.text.secondary} />}
                    label={name.charAt(0).toUpperCase() + name.slice(1)}
                    value={(status as string).toUpperCase()}
                    statusColor={status === 'on' ? theme.colors.success : theme.colors.text.tertiary}
                  />
                  {idx < arr.length - 1 && <View style={styles.divider} />}
                </View>
              ))
            ) : (
              <MetricRow 
                icon={<Power size={20} color={theme.colors.text.secondary} />}
                label="Light / Plug"
                value="No data"
              />
            )}
          </Card>

          {/* Backend UI State */}
          <SectionTitle title="Backend UI State" />
          <Card style={styles.card}>
            {backendStatus?.ui_state ? (
              <>
                <MetricRow 
                  icon={<Monitor size={20} color={theme.colors.text.secondary} />} 
                  label="Screen" 
                  value={backendStatus.ui_state.current_screen} 
                />
                <View style={styles.divider} />
                <MetricRow 
                  icon={<Zap size={20} color={theme.colors.text.secondary} />} 
                  label="Selected" 
                  value={`${backendStatus.ui_state.selected_item} (${backendStatus.ui_state.selected_index})`} 
                />
                <View style={styles.divider} />
                <View style={styles.itemsRow}>
                  {backendStatus.ui_state.items.map((item) => (
                    <View
                      key={item}
                      style={[
                        styles.itemChip,
                        item === backendStatus.ui_state.selected_item && styles.itemChipActive,
                      ]}
                    >
                      <Text style={[
                        styles.itemChipText,
                        item === backendStatus.ui_state.selected_item && styles.itemChipTextActive,
                      ]}>
                        {item}
                      </Text>
                    </View>
                  ))}
                </View>
              </>
            ) : (
              <Text style={styles.noDataText}>No UI state data</Text>
            )}
          </Card>

          {/* Active Mappings */}
          <SectionTitle title="Active Mappings" />
          <Card style={styles.card}>
            {(Object.entries(mappings) as [VisionEventName, string][]).map(([event, action], idx, arr) => (
              <View key={event}>
                <MetricRow
                  icon={<Eye size={20} color={theme.colors.text.secondary} />}
                  label={event.replace(/_/g, ' ')}
                  value={action}
                  statusColor={theme.colors.primaryLight}
                />
                {idx < arr.length - 1 && <View style={styles.divider} />}
              </View>
            ))}
          </Card>

          {/* Hardware (mock metrics) */}
          <SectionTitle title="Hardware Metrics" />
          <Card style={styles.card}>
            <MetricRow 
              icon={<Cpu size={20} color={theme.colors.text.secondary} />} 
              label="CPU Load" 
              value={`${systemMetrics.cpuLoad}%`} 
              statusColor={systemMetrics.cpuLoad > 80 ? theme.colors.warning : theme.colors.success}
            />
            <View style={styles.progressContainer}>
               <View style={[styles.progressBar, { width: `${systemMetrics.cpuLoad}%` }]} />
            </View>
            <View style={styles.divider} />
            <MetricRow 
              icon={<HardDrive size={20} color={theme.colors.text.secondary} />} 
              label="Memory Usage" 
              value={`${systemMetrics.memoryUsage}%`} 
              statusColor={systemMetrics.memoryUsage > 80 ? theme.colors.warning : theme.colors.success}
            />
            <View style={styles.progressContainer}>
               <View style={[styles.progressBar, { width: `${systemMetrics.memoryUsage}%` }]} />
            </View>
          </Card>
          
          <View style={{height: 40}} />
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
  titleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  refreshBtn: {
    padding: theme.spacing.sm,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.md,
  },
  card: {
    marginBottom: theme.spacing.md,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
  },
  metricLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  metricLabel: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
    marginLeft: theme.spacing.md,
  },
  metricValue: {
    ...theme.typography.subtitle,
    flexShrink: 1,
    textAlign: 'right',
  },
  divider: {
    height: 1,
    backgroundColor: theme.colors.surfaceHighlight,
    marginVertical: theme.spacing.sm,
  },
  progressContainer: {
    height: 4,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: 2,
    marginTop: theme.spacing.xs,
    marginBottom: theme.spacing.sm,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: theme.colors.primary,
    borderRadius: 2,
  },
  noDataText: {
    ...theme.typography.body,
    color: theme.colors.text.tertiary,
    fontStyle: 'italic',
    paddingVertical: theme.spacing.sm,
  },
  timestampText: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
    marginTop: theme.spacing.xs,
  },
  itemsRow: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.sm,
  },
  itemChip: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  itemChipActive: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.focusBackground,
  },
  itemChipText: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  itemChipTextActive: {
    color: theme.colors.primary,
    fontWeight: '700',
  },
});
