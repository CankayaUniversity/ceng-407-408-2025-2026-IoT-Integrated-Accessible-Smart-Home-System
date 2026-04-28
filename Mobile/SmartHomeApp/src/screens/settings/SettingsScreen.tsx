import React from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, TouchableOpacity } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { ModeSwitcher } from '../../components/controls/ModeSwitcher';
import { theme } from '../../theme';
import { Settings, HardDrive, Smartphone, Server, Eye, Activity, Sliders } from 'lucide-react-native';
import { useNavigation } from '@react-navigation/native';
import { API_BASE_URL } from '../../config/api';
import { ControlMode } from '../../types';

type SubMenuItemProps = {
  title: string;
  icon: any;
  onPress: () => void;
  subtitle?: string;
}

const SubMenuItem = ({ title, icon, onPress, subtitle }: SubMenuItemProps) => (
  <TouchableOpacity style={styles.subItem} onPress={onPress}>
    <View style={styles.rowLeft}>
      {icon}
      <View style={{marginLeft: theme.spacing.md}}>
        <Text style={styles.itemTitle}>{title}</Text>
        {subtitle && <Text style={styles.itemSubtitle}>{subtitle}</Text>}
      </View>
    </View>
  </TouchableOpacity>
);

export const SettingsScreen = () => {
  const { appMode, controlMode, supportedControlModes, updateControlModeOnBackend } = useStore();
  const navigation = useNavigation<any>();

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Settings" />
        
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* App Mode */}
          <SectionTitle title="App Mode" />
          <ModeSwitcher />

          {/* Control Mode */}
          <SectionTitle title="Control Mode" />
          <Card style={[styles.card, styles.controlModeCard]}>
            {supportedControlModes.map((mode) => (
              <TouchableOpacity
                key={mode}
                style={[styles.controlModeBtn, controlMode === mode && styles.controlModeBtnActive]}
                onPress={() => updateControlModeOnBackend(mode as ControlMode)}
              >
                <Text style={[styles.controlModeText, controlMode === mode && styles.controlModeTextActive]}>
                  {mode === 'eye_only' ? 'Eye Only' : mode === 'hand_only' ? 'Hand Only' : 'Hybrid'}
                </Text>
              </TouchableOpacity>
            ))}
          </Card>

          {/* Accessibility & Mapping */}
          <SectionTitle title="Accessibility" />
          <Card style={styles.card}>
            <SubMenuItem 
              title="Mapping & Diagnostics" 
              subtitle="Customize eye-to-action mapping" 
              icon={<Eye color={theme.colors.primary} size={24} />} 
              onPress={() => navigation.navigate('Gestures')}
            />
            <View style={styles.divider} />
            <SubMenuItem 
              title="System Status" 
              subtitle="Backend metrics & device state" 
              icon={<Activity color={theme.colors.text.secondary} size={24} />} 
              onPress={() => navigation.navigate('System')}
            />
          </Card>

          {/* Connection */}
          <SectionTitle title="Connection" />
          <Card style={styles.card}>
            <SubMenuItem 
              title="FastAPI Server Target" 
              subtitle={API_BASE_URL} 
              icon={<Server color={theme.colors.primary} size={24} />} 
              onPress={() => {}}
            />
          </Card>

          {/* About */}
          <SectionTitle title="About" />
          <Card style={styles.card}>
            <SubMenuItem 
              title="Device Settings" 
              subtitle="Hardware configuration" 
              icon={<HardDrive color={theme.colors.text.secondary} size={24} />} 
              onPress={() => navigation.navigate('System')}
            />
            <View style={styles.divider} />
            <Text style={styles.versionLabel}>Version 1.0.0 (Beta)</Text>
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
  card: {
    padding: theme.spacing.md,
  },
  subItem: {
    paddingVertical: theme.spacing.sm,
  },
  rowLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  itemTitle: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
  },
  itemSubtitle: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  divider: {
    height: 1,
    backgroundColor: theme.colors.surfaceHighlight,
    marginVertical: theme.spacing.md,
  },
  versionLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
    textAlign: 'center',
    paddingVertical: theme.spacing.sm,
  },
  controlModeCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: theme.spacing.xs,
  },
  controlModeBtn: {
    flex: 1,
    paddingVertical: theme.spacing.md,
    alignItems: 'center',
    borderRadius: theme.radius.md,
  },
  controlModeBtnActive: {
    backgroundColor: theme.colors.primaryDark,
  },
  controlModeText: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
  },
  controlModeTextActive: {
    color: theme.colors.text.primary,
    fontWeight: 'bold',
  }
});
