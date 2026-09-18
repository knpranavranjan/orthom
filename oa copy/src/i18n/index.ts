import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from './locales/en.json';
import hi from './locales/hi.json';

export const LANGS = [
  { code: 'en', label: 'EN', native: 'English' },
  { code: 'hi', label: 'हि', native: 'हिन्दी' },
] as const;

export type LangCode = (typeof LANGS)[number]['code'];

void i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, hi: { translation: hi } },
  lng: 'en',
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  // Clinical numerals stay Latin in both languages: degrees, seconds,
  // the 0-10 score and the KL grade. Devanagari digits invite misreading.
  returnNull: false,
});

export function applyLang(code: string): void {
  void i18n.changeLanguage(code);
  document.documentElement.lang = code;
}

/** Flat key->string map for the exported report. */
export function allStrings(): Record<string, string> {
  const src = (i18n.language === 'hi' ? hi : en) as Record<string, string>;
  return { ...(en as Record<string, string>), ...src };
}

export default i18n;
