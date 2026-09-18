import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.zyniw.oascreening',
  appName: 'OA Screening',
  webDir: 'dist',
  server: { androidScheme: 'https' },
};

export default config;
