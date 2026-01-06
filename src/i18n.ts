import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import zhJSON from './locales/zh.json';
import enJSON from './locales/en.json';

// Retrieve language from localStorage or default to 'zh'
const savedLanguage = localStorage.getItem('i18nextLng') || 'zh';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      zh: { translation: zhJSON },
      en: { translation: enJSON },
    },
    lng: savedLanguage, // Default language
    fallbackLng: 'zh',
    interpolation: {
      escapeValue: false, // React already escapes by default
    },
  });

export default i18n;