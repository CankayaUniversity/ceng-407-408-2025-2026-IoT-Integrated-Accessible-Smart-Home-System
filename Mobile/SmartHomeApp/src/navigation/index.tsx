import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { NavigationContainer } from '@react-navigation/native';
import { theme } from '../theme';
import { Home, Thermometer, Shield, Settings } from 'lucide-react-native';

// Screens
import { DashboardScreen } from '../screens/dashboard/DashboardScreen';
import { RoomsScreen } from '../screens/rooms/RoomsScreen';
import { ClimateScreen } from '../screens/climate/ClimateScreen';
import { SecurityScreen } from '../screens/security/SecurityScreen';
import { SettingsScreen } from '../screens/settings/SettingsScreen';
import { SystemStatusScreen } from '../screens/system/SystemStatusScreen';
import { GestureControlScreen } from '../screens/gestures/GestureControlScreen';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const MainTabs = () => {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopWidth: 1,
          borderTopColor: theme.colors.surfaceHighlight,
          paddingBottom: 20,
          paddingTop: 10,
          height: 80,
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.text.secondary,
        tabBarShowLabel: false,
      }}
    >
      <Tab.Screen 
        name="DashboardTab" 
        component={DashboardScreen} 
        options={{
          tabBarIcon: ({ color, size }) => <Home color={color} size={size} />
        }}
      />
      <Tab.Screen 
        name="Climate" 
        component={ClimateScreen} 
        options={{
          tabBarIcon: ({ color, size }) => <Thermometer color={color} size={size} />
        }}
      />
      <Tab.Screen 
        name="Security" 
        component={SecurityScreen} 
        options={{
          tabBarIcon: ({ color, size }) => <Shield color={color} size={size} />
        }}
      />
      <Tab.Screen 
        name="SettingsTab" 
        component={SettingsScreen} 
        options={{
          tabBarIcon: ({ color, size }) => <Settings color={color} size={size} />
        }}
      />
    </Tab.Navigator>
  );
};

export const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: theme.colors.background }
        }}
      >
        <Stack.Screen name="Main" component={MainTabs} />
        
        {/* Detail/Stack Screens */}
        <Stack.Screen 
          name="Rooms" 
          component={RoomsScreen} 
          options={{ 
            headerShown: true, 
            title: "Rooms", 
            headerStyle: { backgroundColor: theme.colors.surface },
            headerTintColor: theme.colors.text.primary,
          }}
        />
        <Stack.Screen 
          name="System" 
          component={SystemStatusScreen} 
          options={{ 
            headerShown: true, 
            title: "System Status", 
            headerStyle: { backgroundColor: theme.colors.surface },
            headerTintColor: theme.colors.text.primary,
          }}
        />
        <Stack.Screen 
          name="Gestures" 
          component={GestureControlScreen} 
          options={{ 
            headerShown: true, 
            title: "Gestures Control", 
            headerStyle: { backgroundColor: theme.colors.surface },
            headerTintColor: theme.colors.text.primary,
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};
