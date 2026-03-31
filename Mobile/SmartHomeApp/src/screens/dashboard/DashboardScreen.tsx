import React from 'react';
import { View, StyleSheet, ScrollView, SafeAreaView } from 'react-native';
import { Header } from '../../components/common/Header';
import { ModeSwitcher } from '../../components/controls/ModeSwitcher';
import { theme } from '../../theme';
import { useStore } from '../../store/useStore';
import { DashboardStandard } from './DashboardStandard';
import { DashboardAccessible } from './DashboardAccessible';

export const DashboardScreen = () => {
  const { appMode } = useStore();

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Header />
        <View style={styles.content}>
          <ModeSwitcher />
          
          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
            {appMode === 'standard' ? <DashboardStandard /> : <DashboardAccessible />}
          </ScrollView>
        </View>
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
  },
  content: {
    flex: 1,
    paddingHorizontal: theme.spacing.lg,
  },
  scrollContent: {
    paddingBottom: theme.spacing.xxl * 2,
  }
});
