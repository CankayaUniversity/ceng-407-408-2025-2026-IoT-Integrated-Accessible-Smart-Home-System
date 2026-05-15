import React from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, TouchableOpacity } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { Thermometer, Droplets, Wind, Plus, Minus } from 'lucide-react-native';

export const ClimateScreen = () => {
  const { climate, setTargetTemperature, appMode } = useStore();
  const isAccessible = appMode === 'accessibility';

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Climate Control" />
        
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Main Temp Control */}
          <Card style={styles.mainCard}>
            <Text style={styles.modeText}>{climate.mode.toUpperCase()} MODE</Text>
            
            <View style={styles.dialContainer}>
              <TouchableOpacity
                style={[styles.circleBtn, isAccessible && styles.circleBtnLarge]}
                onPress={() => setTargetTemperature(climate.targetTemperature - 1)}
              >
                <Minus color={theme.colors.primary} size={isAccessible ? 40 : 32} />
              </TouchableOpacity>
              
              <View style={styles.tempDisplay}>
                <Text style={[styles.humongousText, isAccessible && styles.humongousTextLarge]}>
                  {climate.targetTemperature}°
                </Text>
                <Text style={styles.targetLabel}>Target</Text>
              </View>

              <TouchableOpacity
                style={[styles.circleBtn, isAccessible && styles.circleBtnLarge]}
                onPress={() => setTargetTemperature(climate.targetTemperature + 1)}
              >
                <Plus color={theme.colors.primary} size={isAccessible ? 40 : 32} />
              </TouchableOpacity>
            </View>
            
            <Text style={styles.currentTemp}>Current: {climate.temperature}° C</Text>
          </Card>

          {/* Atmosphere Info */}
          <SectionTitle title="Atmosphere" />
          <View style={styles.row}>
            <Card style={styles.halfCard}>
              <Droplets color={theme.colors.text.secondary} size={24} />
              <View style={{marginTop: theme.spacing.sm}}>
                <Text style={styles.smallCardValue}>{climate.humidity}%</Text>
                <Text style={styles.smallCardLabel}>Humidity</Text>
              </View>
            </Card>

            <Card style={styles.halfCard}>
              <Wind color={theme.colors.success} size={24} />
              <View style={{marginTop: theme.spacing.sm}}>
                <Text style={styles.smallCardValue}>{climate.airQuality}</Text>
                <Text style={styles.smallCardLabel}>Air Quality</Text>
              </View>
            </Card>
          </View>

          {/* Room Temps — hidden in accessible mode for simplicity */}
          {!isAccessible && (
            <>
              <SectionTitle title="Room By Room" />
              {climate.rooms.map((room, idx) => (
                <Card key={idx} style={styles.roomRow}>
                  <Text style={styles.roomName}>{room.name}</Text>
                  <View style={styles.roomTempBox}>
                    <Text style={styles.roomTemp}>{room.temp}°</Text>
                  </View>
                </Card>
              ))}
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
  mainCard: {
    alignItems: 'center',
    paddingVertical: theme.spacing.xxl,
    backgroundColor: '#1E2333',
  },
  modeText: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    letterSpacing: 2,
    marginBottom: theme.spacing.xl,
  },
  dialContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    paddingHorizontal: theme.spacing.lg,
  },
  circleBtn: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: theme.colors.surfaceHighlight,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: theme.colors.primaryDark,
  },
  circleBtnLarge: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 2,
  },
  tempDisplay: {
    alignItems: 'center',
  },
  humongousText: {
    fontSize: 64,
    fontWeight: '800',
    color: theme.colors.text.primary,
  },
  humongousTextLarge: {
    fontSize: 80,
  },
  targetLabel: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
    marginTop: -8,
  },
  currentTemp: {
    ...theme.typography.subtitle,
    color: theme.colors.text.tertiary,
    marginTop: theme.spacing.xl,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  halfCard: {
    width: '47%',
  },
  smallCardValue: {
    ...theme.typography.h2,
    color: theme.colors.text.primary,
  },
  smallCardLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  roomRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.md,
    marginBottom: theme.spacing.sm,
  },
  roomName: {
    ...theme.typography.bodyLarge,
    color: theme.colors.text.primary,
  },
  roomTempBox: {
    backgroundColor: theme.colors.surfaceHighlight,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.radius.lg,
  },
  roomTemp: {
    ...theme.typography.body,
    color: theme.colors.primary,
    fontWeight: '700',
  }
});
