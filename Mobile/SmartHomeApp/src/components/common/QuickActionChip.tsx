import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { QuickAction } from '../../types';
import { theme } from '../../theme';
import { Sun, Moon, LogOut, Film } from 'lucide-react-native';

const getIcon = (iconName: string, color: string) => {
  switch (iconName) {
    case 'sun': return <Sun size={20} color={color} />;
    case 'moon': return <Moon size={20} color={color} />;
    case 'log-out': return <LogOut size={20} color={color} />;
    case 'film': return <Film size={20} color={color} />;
    default: return <Sun size={20} color={color} />;
  }
};

interface QuickActionChipProps {
  action: QuickAction;
  onPress: (id: string) => void;
}

export const QuickActionChip = ({ action, onPress }: QuickActionChipProps) => {
  return (
    <TouchableOpacity 
      style={styles.container}
      onPress={() => onPress(action.id)}
      activeOpacity={0.8}
    >
      <View style={styles.iconContainer}>
        {getIcon(action.iconName, theme.colors.primary)}
      </View>
      <Text style={styles.label}>{action.label}</Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.xl,
    paddingVertical: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.surfaceHighlight,
  },
  iconContainer: {
    marginRight: theme.spacing.sm,
  },
  label: {
    ...theme.typography.caption,
    color: theme.colors.text.primary,
  }
});
