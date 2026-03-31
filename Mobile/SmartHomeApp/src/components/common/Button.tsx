import React from 'react';
import { TouchableOpacity, Text, StyleSheet, TouchableOpacityProps, StyleProp, ViewStyle, TextStyle } from 'react-native';
import { theme } from '../../theme';

interface ButtonProps extends TouchableOpacityProps {
  title: string;
  variant?: 'primary' | 'secondary' | 'outline';
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
  large?: boolean;
}

export const Button = ({ title, variant = 'primary', style, textStyle, large, ...props }: ButtonProps) => {
  const isPrimary = variant === 'primary';
  const isOutline = variant === 'outline';

  let bgColor = theme.colors.surfaceHighlight;
  let textColor = theme.colors.text.primary;
  let borderColor = 'transparent';

  if (isPrimary) {
    bgColor = theme.colors.primary;
    textColor = theme.colors.text.inverse;
  } else if (isOutline) {
    bgColor = 'transparent';
    borderColor = theme.colors.border;
  }

  return (
    <TouchableOpacity 
      style={[
        styles.container, 
        { 
          backgroundColor: bgColor, 
          borderColor: borderColor, 
          borderWidth: isOutline ? 1 : 0,
          paddingVertical: large ? theme.spacing.lg : theme.spacing.md 
        },
        style
      ]} 
      {...props}
    >
      <Text style={[styles.text, { color: textColor }, textStyle]}>
        {title}
      </Text>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: theme.radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: theme.spacing.lg,
  },
  text: {
    ...theme.typography.subtitle,
  },
});
