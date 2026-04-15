import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Modal, FlatList } from 'react-native';
import { theme } from '../../theme';
import { Eye, ChevronDown, Check } from 'lucide-react-native';

interface MappingRowProps {
  eventName: string;
  currentAction: string;
  availableActions: string[];
  onActionChange: (action: string) => void;
}

const EVENT_LABELS: Record<string, string> = {
  look_left: '👁 Look Left',
  look_right: '👁 Look Right',
  short_blink: '✨ Short Blink',
  long_blink: '💤 Long Blink',
};

export const MappingRow = ({ eventName, currentAction, availableActions, onActionChange }: MappingRowProps) => {
  const [showPicker, setShowPicker] = useState(false);

  return (
    <>
      <View style={styles.container}>
        <View style={styles.eventSide}>
          <Eye size={18} color={theme.colors.primary} />
          <Text style={styles.eventLabel}>{EVENT_LABELS[eventName] || eventName}</Text>
        </View>

        <TouchableOpacity style={styles.actionPicker} onPress={() => setShowPicker(true)}>
          <Text style={styles.actionText}>{currentAction}</Text>
          <ChevronDown size={16} color={theme.colors.text.secondary} />
        </TouchableOpacity>
      </View>

      <Modal visible={showPicker} transparent animationType="fade" onRequestClose={() => setShowPicker(false)}>
        <TouchableOpacity style={styles.modalOverlay} activeOpacity={1} onPress={() => setShowPicker(false)}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Action for {EVENT_LABELS[eventName] || eventName}</Text>
            <FlatList
              data={availableActions}
              keyExtractor={(item) => item}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[styles.optionRow, item === currentAction && styles.optionRowSelected]}
                  onPress={() => { onActionChange(item); setShowPicker(false); }}
                >
                  <Text style={[styles.optionText, item === currentAction && styles.optionTextSelected]}>
                    {item}
                  </Text>
                  {item === currentAction && <Check size={18} color={theme.colors.primary} />}
                </TouchableOpacity>
              )}
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surfaceHighlight,
  },
  eventSide: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    flex: 1,
  },
  eventLabel: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
  },
  actionPicker: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.surfaceHighlight,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: theme.colors.border,
    gap: theme.spacing.xs,
  },
  actionText: {
    ...theme.typography.body,
    color: theme.colors.primaryLight,
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
  },
  modalContent: {
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius.xl,
    width: '80%',
    maxHeight: '60%',
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.surfaceHighlight,
  },
  modalTitle: {
    ...theme.typography.h3,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.lg,
  },
  optionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surfaceHighlight,
  },
  optionRowSelected: {
    backgroundColor: theme.colors.focusBackground,
    borderRadius: theme.radius.md,
  },
  optionText: {
    ...theme.typography.subtitle,
    color: theme.colors.text.secondary,
  },
  optionTextSelected: {
    color: theme.colors.primary,
    fontWeight: '700',
  },
});
