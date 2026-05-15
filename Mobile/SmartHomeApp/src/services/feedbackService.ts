import * as Speech from 'expo-speech';
import { Feedback } from '../types';

let lastMessage = '';
let lastSpokenAt = 0;
const COOLDOWN_MS = 2000;

export const speakFeedback = (feedback: Feedback | undefined | null, isEnabled: boolean) => {
  if (!isEnabled || !feedback || !feedback.speak || !feedback.message) {
    return;
  }

  const now = Date.now();
  if (feedback.message === lastMessage && (now - lastSpokenAt) < COOLDOWN_MS) {
    return; // Prevent duplicate immediate repetitions
  }

  Speech.stop();
  Speech.speak(feedback.message, { language: 'tr' });

  lastMessage = feedback.message;
  lastSpokenAt = now;
};
