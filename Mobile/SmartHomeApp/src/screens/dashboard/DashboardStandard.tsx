import React from 'react';
import { View, StyleSheet, ScrollView, Text } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { QuickActionChip } from '../../components/common/QuickActionChip';
import { DeviceCard } from '../../components/cards/DeviceCard';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { Wifi, Activity } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';

export const DashboardStandard = () => {
  const { devices, quickActions, systemMetrics, toggleDeviceAsync } = useStore();
  const navigation = useNavigation<any>();

  return (
    <View style={styles.container}>
      {/* System Status Summary */}
      <Card style={styles.statusCard}>
        <View style={styles.statusRow}>
          <Wifi size={20} color={systemMetrics.apiStatus === 'online' ? theme.colors.success : theme.colors.warning} />
          <Text style={styles.statusText}>API Connected</Text>
        </View>
        <Text style={styles.subText}>Last cmd: {systemMetrics.lastCommand}</Text>
      </Card>

      {/* Quick Actions */}
      <SectionTitle title="Quick Actions" />
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickActions}>
        {quickActions.map(action => (
          <QuickActionChip key={action.id} action={action} onPress={() => {}} />
        ))}
      </ScrollView>

      {/* Devices */}
      <SectionTitle title="My Devices" actionText="View All Rooms" onActionPress={() => navigation.navigate('Rooms')} />
      <View style={styles.deviceGrid}>
        {devices.map(device => (
          <DeviceCard key={device.id} device={device} onToggle={toggleDeviceAsync} />
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: theme.spacing.md,
  },
  statusCard: {
    flexDirection: 'column',
    justifyContent: 'center',
    marginBottom: theme.spacing.lg,
    backgroundColor: '#1E2333', 
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  statusText: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
    marginLeft: theme.spacing.sm,
  },
  subText: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  quickActions: {
    flexDirection: 'row',
    marginBottom: theme.spacing.lg,
  },
  deviceGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  }
});
