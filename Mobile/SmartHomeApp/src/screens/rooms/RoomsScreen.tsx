import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, Text, SafeAreaView, TouchableOpacity } from 'react-native';
import { useStore } from '../../store/useStore';
import { RoomCard } from '../../components/cards/RoomCard';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { theme } from '../../theme';

export const RoomsScreen = () => {
  const { rooms, appMode } = useStore();
  const [filter, setFilter] = useState<'all' | 'indoor' | 'outdoor'>('all');
  const isAccessible = appMode === 'accessibility';

  const filteredRooms = rooms.filter(
    (room) => filter === 'all' || room.category === filter
  );

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Manage Rooms" />
        
        {/* Filters */}
        <View style={styles.filters}>
          {(['all', 'indoor', 'outdoor'] as const).map((cat) => (
            <TouchableOpacity 
              key={cat}
              style={[
                styles.filterChip,
                filter === cat && styles.activeChip,
                isAccessible && styles.filterChipLarge,
              ]}
              onPress={() => setFilter(cat)}
            >
              <Text style={[
                styles.filterText,
                filter === cat && styles.activeFilterText,
                isAccessible && styles.filterTextLarge,
              ]}>
                {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <ScrollView showsVerticalScrollIndicator={false}>
          <View style={[styles.grid, isAccessible && styles.gridAccessible]}>
            {filteredRooms.map((room) => (
              <RoomCard key={room.id} room={room} onPress={() => {}} />
            ))}
          </View>
        </ScrollView>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  container: {
    flex: 1,
    paddingHorizontal: theme.spacing.lg,
    paddingTop: theme.spacing.xl,
  },
  filters: {
    flexDirection: 'row',
    marginBottom: theme.spacing.lg,
  },
  filterChip: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.radius.round,
    backgroundColor: theme.colors.surface,
    marginRight: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.surfaceHighlight,
  },
  filterChipLarge: {
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
  },
  activeChip: {
    backgroundColor: theme.colors.primary,
    borderColor: theme.colors.primary,
  },
  filterText: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  filterTextLarge: {
    ...theme.typography.subtitle,
  },
  activeFilterText: {
    color: theme.colors.text.inverse,
    fontWeight: '600',
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    paddingBottom: theme.spacing.xxl,
  },
  gridAccessible: {
    gap: theme.spacing.md,
  },
});
