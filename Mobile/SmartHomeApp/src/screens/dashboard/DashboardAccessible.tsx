import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { useStore } from '../../store/useStore';
import { DeviceCard } from '../../components/cards/DeviceCard';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { CloudRain } from 'lucide-react-native';

export const DashboardAccessible = () => {
  const { devices, climate, toggleDevice } = useStore();

  return (
    <View style={styles.container}>
      {/* High Visibility Atmosphere Card */}
      <Card style={styles.climateCard}>
        <View style={styles.climateHeader}>
          <CloudRain size={40} color={theme.colors.primary} />
          <Text style={styles.hugeText}>{climate.temperature}°</Text>
        </View>
        <Text style={styles.climateSub}>Air quality is good ({climate.airQuality} AQI)</Text>
      </Card>

      {/* Accessible Device Cards */}
      <View style={styles.deviceList}>
        {devices.map(device => (
          <DeviceCard 
            key={device.id} 
            device={device} 
            onToggle={toggleDevice} 
            accessibleMode 
          />
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: theme.spacing.md,
  },
  climateCard: {
    marginBottom: theme.spacing.xl,
    padding: theme.spacing.xxl,
    backgroundColor: '#1E2333',
    alignItems: 'center',
  },
  climateHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
  },
  hugeText: {
    fontSize: 48,
    fontWeight: '800',
    color: theme.colors.text.primary,
    marginLeft: theme.spacing.lg,
  },
  climateSub: {
    ...theme.typography.bodyLarge,
    color: theme.colors.text.secondary,
    textAlign: 'center',
  },
  deviceList: {
    flexDirection: 'column',
    gap: theme.spacing.md,
  }
});
