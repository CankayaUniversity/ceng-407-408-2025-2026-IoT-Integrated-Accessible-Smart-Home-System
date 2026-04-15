import React from 'react';
import { View, StyleSheet, StyleProp, ViewStyle, TouchableOpacity } from 'react-native';
import { theme } from '../../theme';

interface CardProps {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  onPress?: () => void;
  active?: boolean;
  focused?: boolean;
}

export const Card = ({ children, style, onPress, active, focused }: CardProps) => {
  const Container = onPress ? TouchableOpacity : View;
  
  return (
    <Container 
      style={[
        styles.container, 
        active && styles.active,
        focused && styles.focused,
        style
      ]} 
      onPress={onPress}
      activeOpacity={0.8}
    >
      {children}
    </Container>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.xl,
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.surfaceHighlight,
  },
  active: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.surfaceHighlight,
  },
  focused: {
    borderColor: theme.colors.focusRing,
    borderWidth: theme.accessibility.focusBorderWidth,
    backgroundColor: theme.colors.focusBackground,
  },
});
