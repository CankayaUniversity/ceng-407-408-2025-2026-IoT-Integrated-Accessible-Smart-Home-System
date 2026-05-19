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

const NAV_RIGHT_EVENTS = new Set(['swipe_left', 'thumbs_up']);
const NAV_LEFT_EVENTS  = new Set(['swipe_right', 'thumbs_down']);

export const DashboardScreen = () => {
  const { appMode, fetchBackendStatus, backendStatus } = useStore();
  const navigation = useNavigation<any>();
  const lastTimestampRef = useRef<string | null>(null);

  useEffect(() => {
    fetchBackendStatus();
  }, []);

  useEffect(() => {
    const interval = setInterval(fetchBackendStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Handle gesture-based tab navigation
  useEffect(() => {
    if (appMode !== 'standard' || !backendStatus?.system_state?.last_vision_event) {
      return;
    }

    const event = backendStatus.system_state.last_vision_event;

    // Deduplicate by timestamp so repeated gestures each fire once
    if (lastTimestampRef.current === event.timestamp) {
      return;
    }
    lastTimestampRef.current = event.timestamp;

    const state = navigation.getState();
    const currentRouteName = state?.routes?.[state.index]?.name;
    const currentTabIndex = TAB_ROUTES.indexOf(currentRouteName);

    if (NAV_RIGHT_EVENTS.has(event.name) && currentTabIndex < TAB_ROUTES.length - 1) {
      navigation.navigate(TAB_ROUTES[currentTabIndex + 1]);
    } else if (NAV_LEFT_EVENTS.has(event.name) && currentTabIndex > 0) {
      navigation.navigate(TAB_ROUTES[currentTabIndex - 1]);
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
