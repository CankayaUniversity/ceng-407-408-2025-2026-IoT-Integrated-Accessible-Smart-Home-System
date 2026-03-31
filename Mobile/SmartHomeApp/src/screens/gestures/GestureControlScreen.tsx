import React from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, TouchableOpacity } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { theme } from '../../theme';
import { Hand, ArrowUp, ArrowDown, RefreshCcw } from 'lucide-react-native';

const getGestureIcon = (iconName: string, color: string) => {
  switch(iconName) {
    case 'arrow-up': return <ArrowUp color={color} size={24} />;
    case 'arrow-down': return <ArrowDown color={color} size={24} />;
    case 'refresh-ccw': return <RefreshCcw color={color} size={24} />;
    default: return <Hand color={color} size={24} />;
  }
}

export const GestureControlScreen = () => {
  const { gestures } = useStore();

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Gesture Control" />
        <Text style={styles.description}>Map your physical movements to common smart home actions.</Text>
        
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{paddingBottom: 40}}>
          {gestures.map(gesture => (
            <Card key={gesture.id} style={styles.card}>
              <View style={styles.headerRow}>
                <View style={styles.iconBox}>
                  {getGestureIcon(gesture.iconName, theme.colors.primary)}
                </View>
                <View style={styles.info}>
                  <Text style={styles.name}>{gesture.name}</Text>
                  <Text style={styles.desc}>{gesture.description}</Text>
                </View>
              </View>
              
              <View style={styles.actionRow}>
                <Text style={styles.actionLabel}>Assigned Action</Text>
                <TouchableOpacity style={styles.dropdownMock}>
                  <Text style={styles.dropdownText}>{gesture.assignedAction}</Text>
                </TouchableOpacity>
              </View>
            </Card>
          ))}

          <SectionTitle title="Sensor Status" />
          <Card style={styles.sensorCard}>
            <View style={styles.sensorRow}>
              <View style={[styles.statusDot, { backgroundColor: theme.colors.success }]} />
              <Text style={styles.sensorText}>Camera Vision Sensor Active</Text>
            </View>
            <View style={styles.sensorRow}>
               <View style={[styles.statusDot, { backgroundColor: theme.colors.success }]} />
               <Text style={styles.sensorText}>Neural Engine Processing</Text>
            </View>
          </Card>

          <Button 
            title="Calibrate Sensors" 
            variant="outline" 
            style={{marginTop: theme.spacing.lg}}
            onPress={() => {}}
          />

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
  card: {
    marginBottom: theme.spacing.md,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  iconBox: {
    width: 48,
    height: 48,
    borderRadius: theme.radius.md,
    backgroundColor: theme.colors.surfaceHighlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: theme.spacing.md,
  },
  info: {
    flex: 1,
  },
  name: {
    ...theme.typography.h3,
    color: theme.colors.text.primary,
  },
  desc: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  actionRow: {
    flexDirection: 'column',
    marginTop: theme.spacing.sm,
  },
  actionLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.xs,
  },
  dropdownMock: {
    backgroundColor: theme.colors.surfaceHighlight,
    padding: theme.spacing.md,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  dropdownText: {
    ...theme.typography.body,
    color: theme.colors.primaryLight,
  },
  sensorCard: {
    backgroundColor: '#1E2333',
  },
  sensorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: theme.spacing.xs,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: theme.spacing.sm,
  },
  sensorText: {
    ...theme.typography.body,
    color: theme.colors.text.primary,
  }
});
