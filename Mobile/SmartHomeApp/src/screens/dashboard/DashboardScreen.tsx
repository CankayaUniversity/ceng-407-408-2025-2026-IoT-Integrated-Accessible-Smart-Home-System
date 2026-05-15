import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, ScrollView, SafeAreaView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Header } from '../../components/common/Header';
import { ModeSwitcher } from '../../components/controls/ModeSwitcher';
import { theme } from '../../theme';
import { useStore } from '../../store/useStore';
import { DashboardStandard } from './DashboardStandard';
import { DashboardAccessible } from './DashboardAccessible';

const TAB_ROUTES = ['DashboardTab', 'Climate', 'Security', 'SettingsTab'];

export const DashboardScreen = () => {
  const { appMode, fetchBackendStatus, backendStatus } = useStore();
  const navigation = useNavigation<any>();
  const lastEventRef = useRef<string | null>(null);

  // Kick off an initial backend status fetch on mount
  useEffect(() => {
    fetchBackendStatus();
  }, []);

  // Poll backend for updates (especially swipe events)
  useEffect(() => {
    const interval = setInterval(fetchBackendStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Handle swipe navigation in standard mode
  useEffect(() => {
    if (appMode !== 'standard' || !backendStatus?.system_state?.last_vision_event) {
      return;
    }

    const event = backendStatus.system_state.last_vision_event;
    const eventKey = `${event.source}:${event.name}`;

    // Only process new events
    if (lastEventRef.current === eventKey) {
      return;
    }
    lastEventRef.current = eventKey;

    // Get current route
    const state = navigation.getState();
    const currentRouteName = state?.routes?.[state.index]?.name;
    const currentTabIndex = TAB_ROUTES.indexOf(currentRouteName);

    if (event.name === 'swipe_left' && currentTabIndex < TAB_ROUTES.length - 1) {
      const nextTab = TAB_ROUTES[currentTabIndex + 1];
      navigation.navigate(nextTab);
    } else if (event.name === 'swipe_right' && currentTabIndex > 0) {
      const prevTab = TAB_ROUTES[currentTabIndex - 1];
      navigation.navigate(prevTab);
    }
  }, [appMode, backendStatus, navigation]);

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
