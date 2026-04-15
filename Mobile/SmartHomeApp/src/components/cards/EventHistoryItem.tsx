import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { theme } from '../../theme';
import { EventHistoryEntry } from '../../types';
import { CheckCircle, XCircle } from 'lucide-react-native';

interface EventHistoryItemProps {
  entry: EventHistoryEntry;
}

export const EventHistoryItem = ({ entry }: EventHistoryItemProps) => {
  return (
    <View style={styles.container}>
      <Text style={styles.time}>{entry.timestamp}</Text>
      <View style={styles.eventGroup}>
        <Text style={styles.eventName}>{entry.eventName}</Text>
        <Text style={styles.arrow}>→</Text>
        <Text style={styles.action}>{entry.mappedAction || 'none'}</Text>
      </View>
      {entry.success ? (
        <CheckCircle size={14} color={theme.colors.success} />
      ) : (
        <XCircle size={14} color={theme.colors.error} />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: theme.spacing.xs,
    paddingHorizontal: theme.spacing.sm,
    gap: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surfaceHighlight,
  },
  time: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
    fontVariant: ['tabular-nums'],
    width: 60,
  },
  eventGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: theme.spacing.xs,
  },
  eventName: {
    ...theme.typography.caption,
    color: theme.colors.text.primary,
    fontWeight: '600',
  },
  arrow: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
  },
  action: {
    ...theme.typography.caption,
    color: theme.colors.primaryLight,
    fontWeight: '600',
  },
});
