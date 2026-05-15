import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { useStore } from '../../store/useStore';
import { theme } from '../../theme';
import { Info, CheckCircle, AlertTriangle, XCircle } from 'lucide-react-native';

export const Toast = () => {
  const { toastMessage, toastSeverity, clearToast } = useStore();
  const translateY = useRef(new Animated.Value(-100)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (toastMessage) {
      Animated.parallel([
        Animated.spring(translateY, {
          toValue: 40,
          friction: 6,
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        })
      ]).start();

      const timeout = setTimeout(() => {
        Animated.parallel([
          Animated.timing(opacity, {
            toValue: 0,
            duration: 300,
            useNativeDriver: true,
          }),
          Animated.timing(translateY, {
            toValue: -100,
            duration: 300,
            useNativeDriver: true,
          })
        ]).start(() => {
          clearToast();
        });
      }, 3000);

      return () => clearTimeout(timeout);
    }
  }, [toastMessage]);

  const animatedStyle = {
    transform: [{ translateY }],
    opacity: opacity,
  };

  if (!toastMessage) return null;

  const getIcon = () => {
    switch(toastSeverity) {
      case 'success': return <CheckCircle size={24} color={theme.colors.success} />;
      case 'warning': return <AlertTriangle size={24} color={theme.colors.warning} />;
      case 'error': return <XCircle size={24} color={theme.colors.error} />;
      default: return <Info size={24} color={theme.colors.primary} />;
    }
  };

  return (
    <Animated.View style={[styles.container, animatedStyle]}>
      <View style={styles.content}>
        {getIcon()}
        <Text style={styles.text}>{toastMessage}</Text>
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: theme.spacing.lg,
    right: theme.spacing.lg,
    zIndex: 9999,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.surfaceHighlight,
    padding: theme.spacing.md,
    borderRadius: theme.radius.lg,
    borderWidth: 1,
    borderColor: theme.colors.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  text: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
    marginLeft: theme.spacing.md,
    flex: 1,
  }
});
