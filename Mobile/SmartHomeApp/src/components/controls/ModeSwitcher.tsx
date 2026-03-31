import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { theme } from '../../theme';
import { useStore } from '../../store/useStore';

export const ModeSwitcher = () => {
  const { appMode, setAppMode } = useStore();
  
  const isStandard = appMode === 'standard';

  return (
    <View style={styles.container}>
      <TouchableOpacity 
        style={[styles.tab, isStandard && styles.activeTab]}
        onPress={() => setAppMode('standard')}
      >
        <Text style={[styles.text, isStandard && styles.activeText]}>Standard</Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={[styles.tab, !isStandard && styles.activeTab]}
        onPress={() => setAppMode('accessibility')}
      >
        <Text style={[styles.text, !isStandard && styles.activeText]}>Accessible</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.xl,
    padding: 4,
    marginVertical: theme.spacing.md,
  },
  tab: {
    flex: 1,
    paddingVertical: theme.spacing.sm,
    alignItems: 'center',
    borderRadius: theme.radius.lg,
  },
  activeTab: {
    backgroundColor: theme.colors.surface,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  text: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  activeText: {
    color: theme.colors.primary,
    fontWeight: '700',
  }
});
