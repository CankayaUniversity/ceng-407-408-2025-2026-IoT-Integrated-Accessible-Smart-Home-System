import React, { useRef } from 'react';
import { View, Text, StyleSheet, Pressable, Animated } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Device } from '../../types';
import { theme } from '../../theme';
import { Lightbulb, Thermometer, Lock, Video, Radio, Power } from 'lucide-react-native';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

interface DeviceCardProps {
  device: Device;
  onToggle: (id: string) => void;
  accessibleMode?: boolean;
}

const getDeviceIcon = (type: string, active: boolean) => {
  const color = active ? theme.colors.primary : theme.colors.text.secondary;
  switch (type) {
    case 'light': return <Lightbulb size={24} color={color} />;
    case 'thermostat': return <Thermometer size={24} color={color} />;
    case 'lock': return <Lock size={24} color={color} />;
    case 'camera': return <Video size={24} color={color} />;
    default: return <Radio size={24} color={color} />;
  }
};

export const DeviceCard = ({ device, onToggle, accessibleMode = false }: DeviceCardProps) => {
  const isActive = device.state === 'on';
  const scale = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => { 
    Animated.spring(scale, {
      toValue: 0.96,
      useNativeDriver: true,
      friction: 5,
    }).start();
  };
  
  const handlePressOut = () => { 
    Animated.spring(scale, {
      toValue: 1,
      useNativeDriver: true,
      friction: 5,
    }).start();
  };

  const animatedStyle = {
    transform: [{ scale }]
  };

  if (accessibleMode) {
    return (
      <AnimatedPressable 
        style={[styles.accContainer, isActive && styles.activeContainer, animatedStyle]}
        onPress={() => onToggle(device.id)}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
      >
        <View style={styles.iconWrapper}>
          {getDeviceIcon(device.type, isActive)}
        </View>
        <Text style={styles.accName}>{device.name}</Text>
        <View style={styles.powerBtn}>
          <Power size={32} color={isActive ? theme.colors.primary : theme.colors.text.secondary} />
        </View>
      </AnimatedPressable>
    );
  }

  return (
    <AnimatedPressable 
      style={[styles.container, animatedStyle, isActive && { borderColor: 'transparent' }]}
      onPress={() => onToggle(device.id)}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
    >
      {isActive ? (
        <LinearGradient
          colors={[theme.colors.surfaceHighlight, '#2A1F1C']} // Subtitle copper-ish gradient
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFillObject}
        />
      ) : null}
      
      <View style={styles.header}>
        {getDeviceIcon(device.type, isActive)}
        <View style={[styles.statusDot, isActive && styles.statusActive]} />
      </View>
      
      <View style={styles.footer}>
        <Text style={styles.name} numberOfLines={1}>{device.name}</Text>
        <Text style={styles.status}>
          {device.state === 'offline' ? 'Offline' : (device.value ? `${device.value}${device.type === 'light' ? '%' : '°'}` : (isActive ? 'On' : 'Off'))}
        </Text>
      </View>
    </AnimatedPressable>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.xl,
    padding: theme.spacing.lg,
    width: '47%',
    aspectRatio: 1,
    borderWidth: 1,
    borderColor: theme.colors.border,
    justifyContent: 'space-between',
    marginBottom: theme.spacing.md,
    overflow: 'hidden',
  },
  activeContainer: {
    borderColor: theme.colors.primaryDark,
    backgroundColor: theme.colors.surfaceHighlight,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.border,
  },
  statusActive: {
    backgroundColor: theme.colors.success,
  },
  footer: {
    marginTop: 'auto',
  },
  name: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
    marginBottom: 4,
  },
  status: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  
  // Accessibility Mode Styles
  accContainer: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.xl,
    padding: theme.spacing.xl,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
    borderWidth: 2,
    borderColor: theme.colors.surfaceHighlight,
  },
  accName: {
    ...theme.typography.h2,
    color: theme.colors.text.primary,
    flex: 1,
    marginLeft: theme.spacing.lg,
  },
  iconWrapper: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: theme.colors.surfaceHighlight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  powerBtn: {
    padding: theme.spacing.sm,
  }
});
