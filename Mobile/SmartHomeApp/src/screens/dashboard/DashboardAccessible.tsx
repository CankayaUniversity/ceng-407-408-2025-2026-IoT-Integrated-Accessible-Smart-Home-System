import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, Text, ActivityIndicator, Pressable, ScrollView, Animated } from 'react-native';
import { useStore } from '../../store/useStore';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { Eye, Power, ArrowLeft, ArrowRight, Zap, RotateCcw, Hand } from 'lucide-react-native';
import { Feedback, VisionEventName } from '../../types';

type FeedbackBanner = { message: string; severity: Feedback['severity'] };

const BANNER_DURATION_MS = 2500;

const SEVERITY_COLORS = (theme: any): Record<Feedback['severity'], string> => ({
  success: theme.colors.success,
  error: theme.colors.error,
  warning: theme.colors.warning,
  info: theme.colors.primary,
});

export const DashboardAccessible = () => {
  const {
    showToast,
    backendStatus,
    isBackendConnected,
    fetchBackendStatus,
    sendVisionEvent,
    mappings,
  } = useStore();

  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [banner, setBanner] = useState<FeedbackBanner | null>(null);

  const bannerTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shownTimestamps = useRef(new Set<string>());

  useEffect(() => {
    const load = async () => {
      await fetchBackendStatus();
      setIsInitialLoad(false);
    };
    load();
    const interval = setInterval(fetchBackendStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Catch real Pi gestures from backend poll
  useEffect(() => {
    const lastAction = backendStatus?.system_state?.last_action;
    const ts = lastAction?.timestamp;
    if (!ts || shownTimestamps.current.has(ts)) return;
    if (lastAction.feedback?.message) {
      shownTimestamps.current.add(ts);
      showBanner(lastAction.feedback.message, lastAction.feedback.severity ?? 'info');
    }
  }, [backendStatus?.system_state?.last_action?.timestamp]);

  const showBanner = (message: string, severity: Feedback['severity'] = 'success') => {
    if (bannerTimeout.current) clearTimeout(bannerTimeout.current);
    setBanner({ message, severity });
    bannerTimeout.current = setTimeout(() => setBanner(null), BANNER_DURATION_MS);
  };

  const handleTestEvent = async (source: string, name: string) => {
    try {
      const response = await sendVisionEvent(source, name);
      if (response?.feedback?.message) {
        const ts = backendStatus?.system_state?.last_action?.timestamp;
        if (ts) shownTimestamps.current.add(ts);
        showBanner(response.feedback.message, response.feedback.severity);
      }
    } catch {
      showToast(`Komut gönderilemedi: ${name}`, 'error');
    }
  };

  if (isInitialLoad && !backendStatus) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Backend'e bağlanılıyor...</Text>
      </View>
    );
  }

  if (!backendStatus) {
    return (
      <View style={[styles.container, styles.centered]}>
        <Text style={styles.errorText}>API bağlantısı yok</Text>
        <Pressable style={styles.retryBtn} onPress={fetchBackendStatus}>
          <RotateCcw size={20} color={theme.colors.text.primary} />
          <Text style={styles.retryText}>Tekrar Dene</Text>
        </Pressable>
      </View>
    );
  }

  const { ui_state, system_state } = backendStatus;

  const eyeButtons: { event: VisionEventName; icon: React.ReactNode; label: string }[] = [
    { event: 'eye:look_left',   icon: <ArrowLeft  color={theme.colors.text.primary} size={28} />, label: 'Sol Bakış'    },
    { event: 'eye:look_right',  icon: <ArrowRight color={theme.colors.text.primary} size={28} />, label: 'Sağ Bakış'   },
    { event: 'eye:short_blink', icon: <Zap        color={theme.colors.text.primary} size={28} />, label: 'Kısa Kırpma' },
    { event: 'eye:long_blink',  icon: <RotateCcw  color={theme.colors.text.primary} size={28} />, label: 'Uzun Kırpma' },
  ];

  const handButtons: { event: VisionEventName; icon: React.ReactNode; label: string }[] = [
    { event: 'hand:swipe_left',      icon: <ArrowLeft  color={theme.colors.text.primary} size={28} />, label: 'Sola Kaydır'  },
    { event: 'hand:swipe_right',     icon: <ArrowRight color={theme.colors.text.primary} size={28} />, label: 'Sağa Kaydır' },
    { event: 'hand:pinch',           icon: <Hand       color={theme.colors.text.primary} size={28} />, label: 'Parmak Ucu'  },
    { event: 'hand:open_palm_hold',  icon: <Hand       color={theme.colors.text.primary} size={28} />, label: 'Açık Avuç'   },
  ];

  // Animated banner control
  const bannerAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (banner) {
      Animated.timing(bannerAnim, { toValue: 1, duration: 250, useNativeDriver: true }).start();
    } else {
      Animated.timing(bannerAnim, { toValue: 0, duration: 200, useNativeDriver: true }).start();
    }
  }, [banner, bannerAnim]);

  return (
    <View style={styles.wrapper}>
      <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 120 }}>
        {/* Header */}
        <View style={styles.header}>
          <Eye size={28} color={theme.colors.primary} />
          <Text style={styles.title}>{ui_state.current_screen.replace(/_/g, ' ').toUpperCase()}</Text>
        </View>

        {/* Backend-Driven Menu Items */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.menuScroll}
          style={styles.menuScrollOuter}
        >
          {ui_state.items.map((item) => {
            const isSelected = ui_state.selected_item === item;
            return (
              <Card
                key={item}
                style={[styles.menuItem, isSelected && styles.menuItemSelected]}
                focused={isSelected}
              >
                <Text style={[styles.itemText, isSelected && styles.itemTextSelected]}>
                  {item}
                </Text>
                {isSelected && <Zap size={20} color={theme.colors.primary} />}
              </Card>
            );
          })}
        </ScrollView>

        {/* Active Mappings */}
        <View style={styles.mappingSection}>
          <Text style={styles.sectionTitle}>Aktif Eşlemeler</Text>
          <View style={styles.mappingGrid}>
            {(Object.entries(mappings) as [VisionEventName, string][]).map(([event, action]) => (
              <View key={event} style={styles.mappingChip}>
                <Text style={styles.mappingEvent}>{event.replace(/_/g, ' ')}</Text>
                <Text style={styles.mappingArrow}>→</Text>
                <Text style={styles.mappingAction}>{action}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Device Status */}
        <View style={styles.statusSection}>
          <Text style={styles.sectionTitle}>Cihazlar</Text>
          <View style={styles.deviceRow}>
            {Object.entries(system_state.device_status).map(([name, status]) => (
              <Card key={name} style={styles.deviceIndicator}>
                <Text style={styles.deviceLabel}>{name === 'light' ? 'Işık' : name === 'plug' ? 'Priz' : name}</Text>
                <Power
                  size={24}
                  color={status === 'on' ? theme.colors.success : theme.colors.text.secondary}
                />
              </Card>
            ))}
          </View>
        </View>

        {/* Test Buttons */}
        <View style={styles.testSection}>
          <Text style={styles.sectionTitle}>Göz Komutları</Text>
          <View style={[styles.testGrid, { marginBottom: theme.spacing.lg }]}>
            {eyeButtons.map(({ event, icon, label }) => {
              const [source, name] = event.split(':');
              const mappedAction = mappings[event] ?? '—';
              return (
                <Pressable
                  key={event}
                  onPress={() => handleTestEvent(source, name)}
                  style={({ pressed }) => [
                    styles.testButton,
                    pressed && styles.testButtonPressed,
                  ]}
                >
                  {icon}
                  <View style={styles.testBtnLabels}>
                    <Text style={styles.testBtnGesture}>{label}</Text>
                    <Text style={styles.testBtnAction}>{mappedAction}</Text>
                  </View>
                </Pressable>
              );
            })}
          </View>

          <Text style={styles.sectionTitle}>El Komutları</Text>
          <View style={styles.testGrid}>
            {handButtons.map(({ event, icon, label }) => {
              const [source, name] = event.split(':');
              const mappedAction = mappings[event] ?? '—';
              return (
                <Pressable
                  key={event}
                  onPress={() => handleTestEvent(source, name)}
                  style={({ pressed }) => [
                    styles.testButton,
                    pressed && styles.testButtonPressed,
                  ]}
                >
                  {icon}
                  <View style={styles.testBtnLabels}>
                    <Text style={styles.testBtnGesture}>{label}</Text>
                    <Text style={styles.testBtnAction}>{mappedAction}</Text>
                  </View>
                </Pressable>
              );
            })}
          </View>
        </View>
      </ScrollView>

      {/* Feedback Banner */}
      <Animated.View
        pointerEvents={banner ? 'auto' : 'none'}
        style={[
          styles.banner,
          {
            backgroundColor: SEVERITY_COLORS(theme)[banner?.severity ?? 'info'],
            opacity: bannerAnim,
            transform: [
              {
                translateY: (bannerAnim as any).interpolate({ inputRange: [0, 1], outputRange: [24, 0] }),
              },
            ],
          },
        ]}
      >
        {banner && <Text style={styles.bannerText}>{banner.message}</Text>}
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  wrapper: {
    flex: 1,
  },
  container: {
    flex: 1,
    paddingVertical: theme.spacing.md,
  },
  centered: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    ...theme.typography.subtitle,
    color: theme.colors.text.secondary,
    marginTop: theme.spacing.md,
  },
  errorText: {
    ...theme.typography.h2,
    color: theme.colors.error,
  },
  retryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.lg,
    padding: theme.spacing.md,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.lg,
  },
  retryText: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.xl,
    paddingHorizontal: theme.spacing.sm,
  },
  title: {
    ...theme.typography.h1,
    color: theme.colors.text.primary,
    marginLeft: theme.spacing.md,
  },
  menuScrollOuter: {
    marginBottom: theme.spacing.xl,
  },
  menuScroll: {
    paddingHorizontal: theme.spacing.xs,
    gap: theme.spacing.md,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: theme.spacing.xxl,
    paddingVertical: theme.spacing.xl,
    backgroundColor: '#1E2333',
    borderWidth: 2,
    borderColor: 'transparent',
    minWidth: 140,
    minHeight: theme.accessibility.cardMinHeight,
    gap: theme.spacing.sm,
  },
  menuItemSelected: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.focusBackground,
  },
  itemText: {
    fontSize: 22,
    fontWeight: '600',
    color: theme.colors.text.secondary,
  },
  itemTextSelected: {
    color: theme.colors.primary,
    fontSize: 26,
    fontWeight: '700',
  },
  mappingSection: {
    marginBottom: theme.spacing.xl,
  },
  sectionTitle: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.md,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  mappingGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
  },
  mappingChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.xs,
    backgroundColor: theme.colors.surfaceHighlight,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.radius.lg,
  },
  mappingEvent: {
    ...theme.typography.caption,
    color: theme.colors.text.primary,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  mappingArrow: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
  },
  mappingAction: {
    ...theme.typography.caption,
    color: theme.colors.primaryLight,
    fontWeight: '600',
  },
  statusSection: {
    marginBottom: theme.spacing.xxl,
  },
  deviceRow: {
    flexDirection: 'row',
    gap: theme.spacing.md,
  },
  deviceIndicator: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: theme.spacing.lg,
  },
  deviceLabel: {
    ...theme.typography.h3,
    color: theme.colors.text.primary,
  },
  testSection: {
    marginTop: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
    borderTopWidth: 1,
    borderTopColor: theme.colors.border,
  },
  testGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.md,
    justifyContent: 'space-between',
  },
  testButton: {
    width: '47%',
    flexDirection: 'row',
    alignItems: 'center',
    padding: theme.spacing.lg,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.lg,
    gap: theme.spacing.sm,
  },
  testButtonPressed: {
    backgroundColor: theme.colors.surfaceHighlightPressed,
  },
  testBtnLabels: {
    flexDirection: 'column',
    gap: 2,
  },
  testBtnGesture: {
    ...theme.typography.body,
    fontWeight: '600',
    color: theme.colors.text.primary,
  },
  testBtnAction: {
    ...theme.typography.caption,
    color: theme.colors.primaryLight,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  // Feedback banner
  banner: {
    position: 'absolute',
    bottom: 24,
    left: 16,
    right: 16,
    borderRadius: theme.radius.lg,
    paddingVertical: theme.spacing.lg,
    paddingHorizontal: theme.spacing.xl,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  bannerText: {
    ...theme.typography.h3,
    color: '#ffffff',
    fontWeight: '700',
    textAlign: 'center',
  },
});

export default DashboardAccessible;
