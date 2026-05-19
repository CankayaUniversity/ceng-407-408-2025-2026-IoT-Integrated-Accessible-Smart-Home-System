import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, ScrollView,
  TextInput, Pressable, ActivityIndicator, Alert, Animated,
} from 'react-native';
import { theme } from '../../theme';
import { Card } from '../../components/common/Card';
import { UserCheck, UserX, Trash2, Camera, Users } from 'lucide-react-native';
import { API_BASE_URL } from '../../config/api';

type EnrollStatus = 'idle' | 'requested' | 'capturing' | 'training' | 'done' | 'error';

interface EnrollState {
  status: EnrollStatus;
  name: string | null;
  progress: number;
  total: number;
  message: string | null;
}

const POLL_INTERVAL_MS = 1000;

export const UserManagementScreen = () => {
  const [name, setName] = useState('');
  const [users, setUsers] = useState<string[]>([]);
  const [enrollState, setEnrollState] = useState<EnrollState>({
    status: 'idle', name: null, progress: 0, total: 40, message: null,
  });
  const [loadingUsers, setLoadingUsers] = useState(true);
  const [deletingUser, setDeletingUser] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progressAnim = useRef(new Animated.Value(0)).current;

  const isEnrolling = ['requested', 'capturing', 'training'].includes(enrollState.status);

  // Animate progress bar
  useEffect(() => {
    const target = enrollState.total > 0 ? enrollState.progress / enrollState.total : 0;
    Animated.timing(progressAnim, {
      toValue: target, duration: 300, useNativeDriver: false,
    }).start();
  }, [enrollState.progress, enrollState.total]);

  const fetchUsers = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/users`);
      const data = await res.json();
      setUsers(data.users ?? []);
    } catch {
      // ignore
    } finally {
      setLoadingUsers(false);
    }
  };

  const fetchEnrollStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/enroll-status`);
      const data: EnrollState = await res.json();
      setEnrollState(data);

      if (data.status === 'done' || data.status === 'error') {
        stopPolling();
        fetchUsers();
      }
    } catch {
      // ignore
    }
  };

  const startPolling = () => {
    if (pollRef.current) return;
    pollRef.current = setInterval(fetchEnrollStatus, POLL_INTERVAL_MS);
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    fetchUsers();
    fetchEnrollStatus();
    return () => stopPolling();
  }, []);

  const handleEnroll = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      Alert.alert('Hata', 'Lütfen bir isim girin.');
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/enroll`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!res.ok) {
        const err = await res.json();
        Alert.alert('Hata', err.detail ?? 'Kayıt başlatılamadı.');
        return;
      }
      setName('');
      startPolling();
      fetchEnrollStatus();
    } catch {
      Alert.alert('Hata', 'Backend bağlantısı kurulamadı.');
    }
  };

  const handleDelete = (userName: string) => {
    Alert.alert(
      'Kullanıcıyı Sil',
      `"${userName}" silinsin mi? Bu işlem geri alınamaz.`,
      [
        { text: 'İptal', style: 'cancel' },
        {
          text: 'Sil', style: 'destructive',
          onPress: async () => {
            setDeletingUser(userName);
            try {
              const res = await fetch(`${API_BASE_URL}/users/${encodeURIComponent(userName)}`, {
                method: 'DELETE',
              });
              if (!res.ok) {
                const err = await res.json();
                Alert.alert('Hata', err.detail ?? 'Silinemedi.');
              } else {
                fetchUsers();
              }
            } catch {
              Alert.alert('Hata', 'Silme işlemi başarısız.');
            } finally {
              setDeletingUser(null);
            }
          },
        },
      ]
    );
  };

  const statusColor = () => {
    if (enrollState.status === 'done') return theme.colors.success;
    if (enrollState.status === 'error') return theme.colors.error;
    return theme.colors.primary;
  };

  const statusIcon = () => {
    if (enrollState.status === 'done') return <UserCheck size={20} color={theme.colors.success} />;
    if (enrollState.status === 'error') return <UserX size={20} color={theme.colors.error} />;
    return <Camera size={20} color={theme.colors.primary} />;
  };

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 60 }}>
        <Text style={styles.pageTitle}>Kullanıcı Yönetimi</Text>

        {/* Enrollment Form */}
        <Card style={styles.card}>
          <Text style={styles.sectionLabel}>Yeni Kullanıcı Kaydet</Text>
          <Text style={styles.hint}>
            Kayıt sırasında Pi kamerasına bakın. Sistem 40 kare toplayarak yüzünüzü öğrenir.
          </Text>

          <TextInput
            style={styles.input}
            placeholder="İsim girin (örn: Ömer)"
            placeholderTextColor={theme.colors.text.tertiary}
            value={name}
            onChangeText={setName}
            editable={!isEnrolling}
            autoCapitalize="words"
            returnKeyType="done"
          />

          <Pressable
            style={[styles.enrollBtn, isEnrolling && styles.enrollBtnDisabled]}
            onPress={handleEnroll}
            disabled={isEnrolling}
          >
            {isEnrolling
              ? <ActivityIndicator size="small" color="#fff" />
              : <Camera size={20} color="#fff" />}
            <Text style={styles.enrollBtnText}>
              {isEnrolling ? 'Kayıt Devam Ediyor...' : 'Yüz Kaydet'}
            </Text>
          </Pressable>

          {/* Progress Section */}
          {enrollState.status !== 'idle' && (
            <View style={styles.progressSection}>
              <View style={styles.progressHeader}>
                {statusIcon()}
                <Text style={[styles.progressMessage, { color: statusColor() }]}>
                  {enrollState.message ?? ''}
                </Text>
              </View>

              {enrollState.status === 'capturing' && (
                <>
                  <View style={styles.progressBarBg}>
                    <Animated.View
                      style={[
                        styles.progressBarFill,
                        {
                          width: (progressAnim as any).interpolate({
                            inputRange: [0, 1],
                            outputRange: ['0%', '100%'],
                          }),
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.progressCount}>
                    {enrollState.progress} / {enrollState.total} kare
                  </Text>
                </>
              )}

              {enrollState.status === 'training' && (
                <View style={styles.trainingRow}>
                  <ActivityIndicator size="small" color={theme.colors.primary} />
                  <Text style={styles.trainingText}>Model eğitiliyor...</Text>
                </View>
              )}
            </View>
          )}
        </Card>

        {/* Registered Users */}
        <Card style={styles.card}>
          <View style={styles.usersHeader}>
            <Users size={20} color={theme.colors.primary} />
            <Text style={styles.sectionLabel}>Kayıtlı Kullanıcılar</Text>
          </View>

          {loadingUsers ? (
            <ActivityIndicator size="small" color={theme.colors.primary} style={{ marginTop: 12 }} />
          ) : users.length === 0 ? (
            <Text style={styles.emptyText}>Henüz kayıtlı kullanıcı yok.</Text>
          ) : (
            users.map((user) => (
              <View key={user} style={styles.userRow}>
                <View style={styles.userLeft}>
                  <UserCheck size={20} color={theme.colors.success} />
                  <Text style={styles.userName}>{user}</Text>
                </View>
                <Pressable
                  onPress={() => handleDelete(user)}
                  disabled={deletingUser === user}
                  style={styles.deleteBtn}
                >
                  {deletingUser === user
                    ? <ActivityIndicator size="small" color={theme.colors.error} />
                    : <Trash2 size={18} color={theme.colors.error} />}
                </Pressable>
              </View>
            ))
          )}
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: theme.colors.background },
  container: { flex: 1, paddingHorizontal: theme.spacing.lg, paddingTop: theme.spacing.xl },
  pageTitle: {
    ...theme.typography.h1,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.xl,
  },
  card: {
    padding: theme.spacing.lg,
    marginBottom: theme.spacing.lg,
  },
  sectionLabel: {
    ...theme.typography.h3,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
  hint: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.lg,
    lineHeight: 18,
  },
  input: {
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: theme.radius.md,
    padding: theme.spacing.md,
    color: theme.colors.text.primary,
    ...theme.typography.body,
    marginBottom: theme.spacing.md,
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  enrollBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.sm,
    backgroundColor: theme.colors.primary,
    borderRadius: theme.radius.lg,
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.xl,
  },
  enrollBtnDisabled: { opacity: 0.5 },
  enrollBtnText: {
    ...theme.typography.subtitle,
    color: '#fff',
    fontWeight: '700',
  },
  progressSection: {
    marginTop: theme.spacing.lg,
    gap: theme.spacing.sm,
  },
  progressHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  progressMessage: {
    ...theme.typography.body,
    fontWeight: '600',
    flex: 1,
  },
  progressBarBg: {
    height: 8,
    backgroundColor: theme.colors.surfaceHighlight,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: theme.colors.primary,
    borderRadius: 4,
  },
  progressCount: {
    ...theme.typography.caption,
    color: theme.colors.text.secondary,
    textAlign: 'right',
  },
  trainingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  trainingText: {
    ...theme.typography.body,
    color: theme.colors.text.secondary,
  },
  usersHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.md,
  },
  emptyText: {
    ...theme.typography.caption,
    color: theme.colors.text.tertiary,
    textAlign: 'center',
    paddingVertical: theme.spacing.lg,
  },
  userRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: theme.spacing.md,
    borderTopWidth: 1,
    borderTopColor: theme.colors.surfaceHighlight,
  },
  userLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.md,
  },
  userName: {
    ...theme.typography.subtitle,
    color: theme.colors.text.primary,
  },
  deleteBtn: {
    padding: theme.spacing.sm,
  },
});

export default UserManagementScreen;
