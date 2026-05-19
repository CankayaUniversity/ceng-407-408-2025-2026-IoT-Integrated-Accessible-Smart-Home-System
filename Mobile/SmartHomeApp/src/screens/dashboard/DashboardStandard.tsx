import React, { useCallback, useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, Pressable,
  ActivityIndicator, Switch, SafeAreaView,
} from 'react-native';
import { useStore } from '../../store/useStore';
import { theme } from '../../theme';
import { Lightbulb, Plug, Wifi, WifiOff, Clock, Zap, UserCheck } from 'lucide-react-native';
import { executeAction, setLightColor } from '../../services/apiClient';

// ─── Color Presets ───────────────────────────────────────────────────────────

interface ColorPreset {
  label: string;
  color: string;          // display color in UI
  hue?: number;
  saturation?: number;
  color_temperature?: number;
  brightness: number;
}

const COLOR_PRESETS: ColorPreset[] = [
  { label: 'Sıcak',    color: '#FFB347', color_temperature: 2700, brightness: 80 },
  { label: 'Soğuk',    color: '#E8F4FF', color_temperature: 6500, brightness: 100 },
  { label: 'Kırmızı',  color: '#EF4444', hue: 0,   saturation: 100, brightness: 80 },
  { label: 'Turuncu',  color: '#FB923C', hue: 25,  saturation: 100, brightness: 80 },
  { label: 'Yeşil',    color: '#4ADE80', hue: 120, saturation: 100, brightness: 70 },
  { label: 'Mavi',     color: '#60A5FA', hue: 220, saturation: 100, brightness: 80 },
  { label: 'Mor',      color: '#A78BFA', hue: 270, saturation: 80,  brightness: 80 },
];

const BRIGHTNESS_LEVELS = [20, 40, 60, 80, 100];

// ─── Sub-components ──────────────────────────────────────────────────────────

const ConnectionBadge = ({ connected }: { connected: boolean }) => (
  <View style={[styles.badge, { backgroundColor: connected ? theme.colors.success + '22' : theme.colors.error + '22' }]}>
    {connected
      ? <Wifi size={12} color={theme.colors.success} />
      : <WifiOff size={12} color={theme.colors.error} />}
    <Text style={[styles.badgeText, { color: connected ? theme.colors.success : theme.colors.error }]}>
      {connected ? 'Bağlı' : 'Bağlantı Yok'}
    </Text>
  </View>
);

// ─── Main Component ───────────────────────────────────────────────────────────

export const DashboardStandard = () => {
  const { backendStatus, isBackendConnected, fetchBackendStatus } = useStore();

  const [lightLoading, setLightLoading] = useState(false);
  const [plugLoading, setPlugLoading] = useState(false);
  const [colorLoading, setColorLoading] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [activeBrightness, setActiveBrightness] = useState<number | null>(null);

  const deviceStatus = backendStatus?.system_state?.device_status;
  const lightOn = deviceStatus?.light === 'on';
  const plugOn = deviceStatus?.plug === 'on';
  const faceAuth = backendStatus?.system_state?.face_auth;
  const lastAction = backendStatus?.system_state?.last_action;

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return 'Günaydın';
    if (h < 18) return 'İyi günler';
    return 'İyi akşamlar';
  };

  const handleLight = useCallback(async (turnOn: boolean) => {
    setLightLoading(true);
    try {
      await executeAction(turnOn ? 'LIGHT_ON' : 'LIGHT_OFF');
      await fetchBackendStatus();
    } finally {
      setLightLoading(false);
    }
  }, [fetchBackendStatus]);

  const handlePlug = useCallback(async (turnOn: boolean) => {
    setPlugLoading(true);
    try {
      await executeAction(turnOn ? 'PLUG_ON' : 'PLUG_OFF');
      await fetchBackendStatus();
    } finally {
      setPlugLoading(false);
    }
  }, [fetchBackendStatus]);

  const handleColorPreset = useCallback(async (preset: ColorPreset) => {
    setColorLoading(true);
    setActivePreset(preset.label);
    setActiveBrightness(preset.brightness);
    try {
      await setLightColor({
        hue: preset.hue,
        saturation: preset.saturation,
        brightness: preset.brightness,
        color_temperature: preset.color_temperature,
      });
      await fetchBackendStatus();
    } finally {
      setColorLoading(false);
    }
  }, [fetchBackendStatus]);

  const handleBrightness = useCallback(async (level: number) => {
    setColorLoading(true);
    setActiveBrightness(level);
    try {
      await setLightColor({ brightness: level });
      await fetchBackendStatus();
    } finally {
      setColorLoading(false);
    }
  }, [fetchBackendStatus]);

  if (!backendStatus) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Bağlanılıyor...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Header ── */}
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>{greeting()}</Text>
            {faceAuth?.user && (
              <View style={styles.userRow}>
                <UserCheck size={14} color={theme.colors.success} />
                <Text style={styles.userName}>{faceAuth.user}</Text>
              </View>
            )}
          </View>
          <ConnectionBadge connected={isBackendConnected} />
        </View>

        {/* ── Ampul Kartı ── */}
        <View style={styles.card}>
          <View style={styles.deviceHeader}>
            <View style={styles.deviceLeft}>
              <View style={[styles.deviceIcon, { backgroundColor: lightOn ? '#FFB34722' : theme.colors.surfaceHighlight }]}>
                <Lightbulb size={24} color={lightOn ? '#FFB347' : theme.colors.text.tertiary} />
              </View>
              <View>
                <Text style={styles.deviceName}>Akıllı Ampul</Text>
                <Text style={[styles.deviceStatus, { color: lightOn ? theme.colors.success : theme.colors.text.tertiary }]}>
                  {lightOn ? 'Açık' : 'Kapalı'}
                </Text>
              </View>
            </View>
            {lightLoading
              ? <ActivityIndicator size="small" color={theme.colors.primary} />
              : (
                <Switch
                  value={lightOn}
                  onValueChange={handleLight}
                  trackColor={{ false: theme.colors.surfaceHighlight, true: theme.colors.primary }}
                  thumbColor="#fff"
                />
              )}
          </View>

          {/* Renk & Parlaklık — sadece ışık açıkken */}
          {lightOn && (
            <View style={styles.colorPanel}>
              <View style={styles.divider} />

              <Text style={styles.panelLabel}>Renk</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetScroll}>
                {COLOR_PRESETS.map((preset) => (
                  <Pressable
                    key={preset.label}
                    onPress={() => handleColorPreset(preset)}
                    style={[
                      styles.presetChip,
                      activePreset === preset.label && styles.presetChipActive,
                    ]}
                  >
                    <View style={[styles.presetDot, { backgroundColor: preset.color }]} />
                    <Text style={[
                      styles.presetLabel,
                      activePreset === preset.label && { color: theme.colors.text.primary },
                    ]}>
                      {preset.label}
                    </Text>
                  </Pressable>
                ))}
              </ScrollView>

              <Text style={[styles.panelLabel, { marginTop: theme.spacing.md }]}>Parlaklık</Text>
              <View style={styles.brightnessRow}>
                {BRIGHTNESS_LEVELS.map((level) => (
                  <Pressable
                    key={level}
                    onPress={() => handleBrightness(level)}
                    style={[
                      styles.brightnessBtn,
                      activeBrightness === level && styles.brightnessBtnActive,
                    ]}
                  >
                    <Text style={[
                      styles.brightnessBtnText,
                      activeBrightness === level && styles.brightnessBtnTextActive,
                    ]}>
                      {level}%
                    </Text>
                  </Pressable>
                ))}
              </View>

              {colorLoading && (
                <View style={styles.colorLoadingRow}>
                  <ActivityIndicator size="small" color={theme.colors.primary} />
                  <Text style={styles.colorLoadingText}>Uygulanıyor...</Text>
                </View>
              )}
            </View>
          )}
        </View>

        {/* ── Priz Kartı ── */}
        <View style={styles.card}>
          <View style={styles.deviceHeader}>
            <View style={styles.deviceLeft}>
              <View style={[styles.deviceIcon, { backgroundColor: plugOn ? theme.colors.success + '22' : theme.colors.surfaceHighlight }]}>
                <Plug size={24} color={plugOn ? theme.colors.success : theme.colors.text.tertiary} />
              </View>
              <View>
                <Text style={styles.deviceName}>Akıllı Priz</Text>
                <Text style={[styles.deviceStatus, { color: plugOn ? theme.colors.success : theme.colors.text.tertiary }]}>
                  {plugOn ? 'Açık' : 'Kapalı'}
                </Text>
              </View>
            </View>
            {plugLoading
              ? <ActivityIndicator size="small" color={theme.colors.primary} />
              : (
                <Switch
                  value={plugOn}
                  onValueChange={handlePlug}
                  trackColor={{ false: theme.colors.surfaceHighlight, true: theme.colors.success }}
                  thumbColor="#fff"
                />
              )}
          </View>
        </View>

        {/* ── Son Aksiyon ── */}
        {lastAction && (
          <View style={styles.lastActionCard}>
            <View style={styles.lastActionLeft}>
              <Zap size={14} color={theme.colors.primary} />
              <Text style={styles.lastActionLabel}>Son Aksiyon</Text>
            </View>
            <View style={styles.lastActionRight}>
              <Text style={styles.lastActionValue}>{lastAction.action}</Text>
              <View style={styles.lastActionTimeRow}>
                <Clock size={11} color={theme.colors.text.tertiary} />
                <Text style={styles.lastActionTime}>
                  {new Date(lastAction.timestamp).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
                </Text>
              </View>
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

// ─── Styles ──────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: theme.colors.background },
  scroll: { flex: 1 },
  scrollContent: {
    paddingHorizontal: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
    paddingBottom: 60,
    gap: theme.spacing.md,
  },
  centered: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.background,
  },
  loadingText: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
    marginTop: theme.spacing.md,
  },

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.md,
  },
  greeting: {
    ...theme.typography.h2,
    color: theme.colors.text.primary,
  },
  userRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.xs,
    marginTop: 4,
  },
  userName: {
    ...theme.typography.caption,
    color: theme.colors.success,
    fontWeight: '600',
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 5,
    borderRadius: theme.radius.round,
  },
  badgeText: {
    ...theme.typography.caption,
    fontWeight: '600',
  },

  // Device Card
  card: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.lg,
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  deviceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  deviceLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.md,
  },
  deviceIcon: {
    width: 48,
    height: 48,
    borderRadius: theme.radius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  deviceName: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
    fontWeight: '600',
  },
  deviceStatus: {
    ...theme.typography.caption,
    fontWeight: '600',
    marginTop: 2,
  },

  // Color Panel
  colorPanel: { marginTop: theme.spacing.md },
  divider: {
    height: 1,
    backgroundColor: theme.colors.border,
    marginBottom: theme.spacing.md,
  },
  panelLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: theme.spacing.sm,
  },
  presetScroll: { marginHorizontal: -theme.spacing.xs },
  presetChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.radius.round,
    backgroundColor: theme.colors.surfaceHighlight,
    marginRight: theme.spacing.sm,
    borderWidth: 1.5,
    borderColor: 'transparent',
  },
  presetChipActive: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.focusBackground,
  },
  presetDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  presetLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    fontWeight: '600',
  },

  // Brightness
  brightnessRow: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
  },
  brightnessBtn: {
    flex: 1,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.radius.md,
    backgroundColor: theme.colors.surfaceHighlight,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'transparent',
  },
  brightnessBtnActive: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.focusBackground,
  },
  brightnessBtnText: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    fontWeight: '600',
  },
  brightnessBtnTextActive: {
    color: theme.colors.primary,
  },

  // Color loading
  colorLoadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.md,
  },
  colorLoadingText: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },

  // Last action
  lastActionCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.lg,
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  lastActionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.xs,
  },
  lastActionLabel: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  lastActionRight: { alignItems: 'flex-end', gap: 2 },
  lastActionValue: {
    ...theme.typography.body,
    color: theme.colors.primary,
    fontWeight: '700',
  },
  lastActionTimeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
  },
  lastActionTime: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
  },
});

export default DashboardStandard;
