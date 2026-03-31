import React from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, TouchableOpacity } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { ModeSwitcher } from '../../components/controls/ModeSwitcher';
import { theme } from '../../theme';
import { Settings, HelpCircle, HardDrive, Smartphone, Server } from 'lucide-react-native';
import { NavigationProp, useNavigation } from '@react-navigation/native';

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
  const { appMode } = useStore();
  const navigation = useNavigation<any>();

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Settings" />
        
        <ScrollView showsVerticalScrollIndicator={false}>
          <SectionTitle title="App Appearance" />
          <ModeSwitcher />

          <SectionTitle title="Device Settings" />
          <Card style={styles.card}>
            <SubMenuItem 
              title="Gestures Control" 
              subtitle="Map gestures to actions" 
              icon={<Smartphone color={theme.colors.text.secondary} size={24} />} 
              onPress={() => navigation.navigate('Gestures')}
            />
            <View style={styles.divider} />
            <SubMenuItem 
              title="System Status" 
              subtitle="Check hardware state" 
              icon={<HardDrive color={theme.colors.text.secondary} size={24} />} 
              onPress={() => navigation.navigate('System')}
            />
          </Card>

          <SectionTitle title="Connection (API Ready)" />
          <Card style={styles.card}>
            <SubMenuItem 
              title="FastAPI Server Target" 
              subtitle="http://raspberrypi5.local:8000" 
              icon={<Server color={theme.colors.primary} size={24} />} 
              onPress={() => {}}
            />
          </Card>

          <SectionTitle title="About" />
          <Card style={styles.card}>
            <SubMenuItem 
              title="Help & Support" 
              icon={<HelpCircle color={theme.colors.text.secondary} size={24} />} 
              onPress={() => {}}
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
  }
});
