import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, Switch } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { Button } from '../../components/common/Button';
import { theme } from '../../theme';
import { ShieldCheck, ShieldAlert, DoorClosed, LayoutList } from 'lucide-react-native';

export const SecurityScreen = () => {
  const { security } = useStore();
  const [armed, setArmed] = useState(security.armed);
  const [doors, setDoors] = useState(security.doorsLocked);

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="Security Hub" />
        
        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Main Status */}
          <Card style={[styles.mainCard, armed ? styles.cardArmed : styles.cardDisarmed]}>
            <View style={styles.iconCircle}>
              {armed ? (
                <ShieldCheck size={48} color={theme.colors.success} />
              ) : (
                <ShieldAlert size={48} color={theme.colors.warning} />
              )}
            </View>
            <Text style={styles.statusTitle}>
              {armed ? 'System is Armed' : 'System is Disarmed'}
            </Text>
            <Text style={styles.statusDesc}>
              {security.activeAlarms === 0 ? 'No alarms detected in the last 24h.' : `${security.activeAlarms} active alarms!`}
            </Text>

            <Button 
              title={armed ? "Disarm System" : "Arm System"} 
              variant={armed ? "outline" : "primary"}
              large
              style={styles.armBtn}
              textStyle={armed ? { color: theme.colors.success } : {}}
              onPress={() => setArmed(!armed)}
            />
          </Card>

          <SectionTitle title="Access Control" />
          <Card style={styles.accessCard}>
            <View style={styles.row}>
              <View style={styles.rowLeft}>
                <DoorClosed color={theme.colors.text.secondary} size={24} />
                <View style={styles.col}>
                  <Text style={styles.rowTitle}>All Doors Locked</Text>
                  <Text style={styles.rowSub}>Front, Garage, Back door</Text>
                </View>
              </View>
              <Switch 
                value={doors} 
                onValueChange={setDoors} 
                trackColor={{ false: theme.colors.surfaceHighlight, true: theme.colors.primaryDark }}
                thumbColor={doors ? theme.colors.primary : theme.colors.text.tertiary}
              />
            </View>
            <View style={styles.divider} />
            <View style={styles.row}>
              <View style={styles.rowLeft}>
                <LayoutList color={theme.colors.text.secondary} size={24} />
                <View style={styles.col}>
                  <Text style={styles.rowTitle}>Windows Closed</Text>
                  <Text style={styles.rowSub}>2 open on 1st floor</Text>
                </View>
              </View>
              <Switch 
                value={security.windowsClosed} 
                disabled
                trackColor={{ false: theme.colors.surfaceHighlight, true: theme.colors.primaryDark }}
                thumbColor={security.windowsClosed ? theme.colors.primary : theme.colors.warning}
              />
            </View>
          </Card>

          <SectionTitle title="Live Cameras" />
          <Card style={styles.camCard}>
            <View style={styles.camBg}>
              <Text style={styles.camStatus}>Kitchen Cam Offline</Text>
            </View>
            <Text style={styles.camName}>Kitchen</Text>
          </Card>

          <View style={{height: 40}} />
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
  mainCard: {
    alignItems: 'center',
    paddingVertical: theme.spacing.xxl,
    borderWidth: 2,
    marginBottom: theme.spacing.lg,
  },
  cardArmed: {
    borderColor: theme.colors.success,
    backgroundColor: 'rgba(74, 222, 128, 0.05)',
  },
  cardDisarmed: {
    borderColor: theme.colors.warning,
    backgroundColor: 'rgba(251, 146, 60, 0.05)',
  },
  iconCircle: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: theme.colors.surfaceHighlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  statusTitle: {
    ...theme.typography.h2,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.xs,
  },
  statusDesc: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
  },
  armBtn: {
    marginTop: theme.spacing.xl,
    width: '100%',
  },
  accessCard: {
    marginBottom: theme.spacing.lg,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
  },
  rowLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  col: {
    marginLeft: theme.spacing.md,
  },
  rowTitle: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
  },
  rowSub: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
  },
  divider: {
    height: 1,
    backgroundColor: theme.colors.surfaceHighlight,
    marginVertical: theme.spacing.sm,
  },
  camCard: {
    padding: 0,
    overflow: 'hidden',
  },
  camBg: {
    height: 160,
    backgroundColor: theme.colors.surfaceHighlight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  camStatus: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
  },
  camName: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
    padding: theme.spacing.md,
  }
});
