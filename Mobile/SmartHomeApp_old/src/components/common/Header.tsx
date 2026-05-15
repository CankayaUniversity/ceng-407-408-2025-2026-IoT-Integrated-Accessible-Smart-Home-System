import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { theme } from '../../theme';
import { User, Wifi, WifiOff } from 'lucide-react-native';
import { useStore } from '../../store/useStore';

export const Header = () => {
  const { isBackendConnected } = useStore();

  return (
    <View style={styles.container}>
      <View>
        <Text style={styles.greeting}>Welcome Home</Text>
        <Text style={styles.title}>Smart Home</Text>
      </View>
      <View style={styles.rightSection}>
        <View style={styles.connectionBadge}>
          {isBackendConnected ? (
            <Wifi color={theme.colors.success} size={16} />
          ) : (
            <WifiOff color={theme.colors.error} size={16} />
          )}
        </View>
        <TouchableOpacity style={styles.avatar}>
          <User color={theme.colors.primary} size={24} />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
    paddingBottom: theme.spacing.md,
  },
  greeting: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  title: {
    ...theme.typography.h2,
    color: theme.colors.text.primary,
  },
  rightSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  connectionBadge: {
    width: 32,
    height: 32,
    borderRadius: theme.radius.round,
    backgroundColor: theme.colors.surfaceHighlight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: theme.radius.round,
    backgroundColor: theme.colors.surfaceHighlight,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
});
