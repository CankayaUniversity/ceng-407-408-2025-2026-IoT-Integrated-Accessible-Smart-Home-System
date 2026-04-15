import React from 'react';
import { View, StyleSheet, ScrollView, Text } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { QuickActionChip } from '../../components/common/QuickActionChip';
import { DeviceCard } from '../../components/cards/DeviceCard';
import { StatusIndicator } from '../../components/cards/StatusIndicator';
import { theme } from '../../theme';
import { useNavigation } from '@react-navigation/native';

export const DashboardStandard = () => {
  const { devices, quickActions, toggleDeviceAsync, backendStatus, isBackendConnected } = useStore();
  const navigation = useNavigation<any>();

  return (
    <View style={styles.container}>
      {/* Backend Status Summary */}
      <StatusIndicator
        isConnected={isBackendConnected}
        lastEvent={backendStatus?.system_state?.last_vision_event?.name || null}
        lastAction={backendStatus?.system_state?.last_action?.action || null}
        controllerMode={backendStatus?.device_controller_mode}
      />

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
