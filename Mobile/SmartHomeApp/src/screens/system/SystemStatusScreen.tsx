import React from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView } from 'react-native';
import { useStore } from '../../store/useStore';
import { SectionTitle } from '../../components/layout/SectionTitle';
import { Card } from '../../components/common/Card';
import { theme } from '../../theme';
import { Activity, Cpu, HardDrive, Network, Zap } from 'lucide-react-native';

const MetricRow = ({ icon, label, value, statusColor }: any) => (
  <View style={styles.metricRow}>
    <View style={styles.metricLeft}>
      {icon}
      <Text style={styles.metricLabel}>{label}</Text>
    </View>
    <Text style={[styles.metricValue, { color: statusColor || theme.colors.text.primary }]}>
      {value}
    </Text>
  </View>
);

export const SystemStatusScreen = () => {
  const { systemMetrics } = useStore();

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <SectionTitle title="System Status" />
        
        <ScrollView showsVerticalScrollIndicator={false}>
          <Card style={styles.card}>
            <MetricRow 
              icon={<Network size={20} color={theme.colors.text.secondary} />} 
              label="API Status" 
              value={systemMetrics.apiStatus.toUpperCase()} 
              statusColor={systemMetrics.apiStatus === 'online' ? theme.colors.success : theme.colors.warning} 
            />
            <View style={styles.divider} />
            <MetricRow 
              icon={<Activity size={20} color={theme.colors.text.secondary} />} 
              label="Last Event (Gaze/API)" 
              value={systemMetrics.lastEvent} 
            />
            <View style={styles.divider} />
            <MetricRow 
              icon={<Zap size={20} color={theme.colors.text.secondary} />} 
              label="Last Command" 
              value={systemMetrics.lastCommand} 
            />
          </Card>

          <SectionTitle title="Hardware Core" />
          <Card style={styles.card}>
            <MetricRow 
              icon={<Cpu size={20} color={theme.colors.text.secondary} />} 
              label="CPU Load" 
              value={`${systemMetrics.cpuLoad}%`} 
              statusColor={systemMetrics.cpuLoad > 80 ? theme.colors.warning : theme.colors.success}
            />
            
            {/* Simple progress bar mock */}
            <View style={styles.progressContainer}>
               <View style={[styles.progressBar, { width: `${systemMetrics.cpuLoad}%` }]} />
            </View>

            <View style={styles.divider} />
            
            <MetricRow 
              icon={<HardDrive size={20} color={theme.colors.text.secondary} />} 
              label="Memory Usage" 
              value={`${systemMetrics.memoryUsage}%`} 
              statusColor={systemMetrics.memoryUsage > 80 ? theme.colors.warning : theme.colors.success}
            />
            <View style={styles.progressContainer}>
               <View style={[styles.progressBar, { width: `${systemMetrics.memoryUsage}%` }]} />
            </View>
          </Card>

          <SectionTitle title="Engine Info" />
          <Card style={styles.card}>
            <MetricRow 
              icon={<Zap size={20} color={theme.colors.text.secondary} />} 
              label="Neural Engine" 
              value={systemMetrics.neuralEngine} 
            />
            <View style={styles.divider} />
            <MetricRow 
              icon={<Activity size={20} color={theme.colors.text.secondary} />} 
              label="Uptime" 
              value={systemMetrics.uptime} 
            />
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
  card: {
    marginBottom: theme.spacing.md,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
  },
  metricLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  metricLabel: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
    marginLeft: theme.spacing.md,
  },
  metricValue: {
    ...theme.typography.subtitle,
  },
  divider: {
    height: 1,
    backgroundColor: theme.colors.surfaceHighlight,
    marginVertical: theme.spacing.sm,
  },
  progressContainer: {
    height: 4,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: 2,
    marginTop: theme.spacing.xs,
    marginBottom: theme.spacing.sm,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: theme.colors.primary,
    borderRadius: 2,
  }
});
