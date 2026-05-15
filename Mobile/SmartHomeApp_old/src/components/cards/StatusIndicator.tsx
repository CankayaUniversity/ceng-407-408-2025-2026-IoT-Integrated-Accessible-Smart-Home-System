import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { theme } from '../../theme';
import { Wifi, WifiOff, Activity, Zap } from 'lucide-react-native';

interface StatusIndicatorProps {
  isConnected: boolean;
  lastEvent?: string | null;
  lastAction?: string | null;
  controllerMode?: string;
}

export const StatusIndicator = ({ isConnected, lastEvent, lastAction, controllerMode }: StatusIndicatorProps) => {
  return (
    <View style={styles.container}>
      <View style={styles.row}>
        {isConnected ? (
          <Wifi size={16} color={theme.colors.success} />
        ) : (
          <WifiOff size={16} color={theme.colors.error} />
        )}
        <Text style={[styles.statusText, { color: isConnected ? theme.colors.success : theme.colors.error }]}>
          {isConnected ? 'API Connected' : 'Disconnected'}
        </Text>
        {controllerMode && (
          <View style={styles.modeBadge}>
            <Text style={styles.modeText}>{controllerMode}</Text>
          </View>
        )}
      </View>

      {lastEvent && (
        <View style={styles.row}>
          <Activity size={14} color={theme.colors.text.tertiary} />
          <Text style={styles.infoText}>Last event: {lastEvent}</Text>
        </View>
      )}

      {lastAction && (
        <View style={styles.row}>
          <Zap size={14} color={theme.colors.text.tertiary} />
          <Text style={styles.infoText}>Last action: {lastAction}</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.lg,
    padding: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.surfaceHighlight,
    gap: theme.spacing.xs,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  statusText: {
    ...theme.typography.caption,
    fontWeight: '600',
  },
  modeBadge: {
    backgroundColor: theme.colors.surfaceHighlight,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.radius.sm,
    marginLeft: 'auto',
  },
  modeText: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
    fontSize: 10,
  },
  infoText: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
  },
});
