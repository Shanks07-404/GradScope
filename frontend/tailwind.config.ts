import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        chalkboard: {
          bg: '#0E1512',
          card: '#131D19',
          border: '#232D28',
          grid: '#2A332E',
          chalk: '#EDEAE0',
          amber: '#F2B84B',
          teal: '#4C9A8E',
          sage: '#8FA69D',
          dim: '#3F4F48',
        }
      },
      fontFamily: {
        serif: ['Fraunces', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
