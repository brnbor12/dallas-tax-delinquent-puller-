/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        hot:  '#ef4444',   // red-500
        warm: '#f97316',   // orange-500
        cold: '#3b82f6',   // blue-500
      },
    },
  },
  plugins: [],
}
