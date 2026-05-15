import React from 'react';
import { View, Text, StyleSheet, ImageBackground, TouchableOpacity } from 'react-native';
import { Room } from '../../types';
import { theme } from '../../theme';
import { ChevronRight } from 'lucide-react-native';

interface RoomCardProps {
  room: Room;
  onPress: (id: string) => void;
}

export const RoomCard = ({ room, onPress }: RoomCardProps) => {
  return (
    <TouchableOpacity style={styles.container} onPress={() => onPress(room.id)} activeOpacity={0.8}>
      <View style={styles.placeholderBg}>
        <View style={styles.overlay}>
          <View>
            <Text style={styles.name}>{room.name}</Text>
            <Text style={styles.deviceCount}>{room.devicesCount} Devices</Text>
          </View>
          <View style={styles.arrowContainer}>
             <ChevronRight color={theme.colors.text.primary} size={20} />
          </View>
        </View>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '47%',
    aspectRatio: 1,
    borderRadius: theme.radius.xl,
    overflow: 'hidden',
    marginBottom: theme.spacing.md,
    backgroundColor: theme.colors.surfaceHighlight,
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  placeholderBg: {
    flex: 1,
    backgroundColor: theme.colors.surfaceHighlight, // In real app, this would be an ImageBackground
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    padding: theme.spacing.md,
    justifyContent: 'space-between',
  },
  name: {
    ...theme.typography.h3,
    color: '#FFFFFF',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: {width: -1, height: 1},
    textShadowRadius: 10,
  },
  deviceCount: {
    ...theme.typography.caption,
    color: '#E2E8F0',
  },
  arrowContainer: {
    alignSelf: 'flex-end',
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: theme.radius.round,
    padding: theme.spacing.xs,
  }
});
