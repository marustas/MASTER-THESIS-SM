import { createTheme } from "@mui/material/styles";

// VILNIUS TECH official palette (sourced from vilniustech.lt CSS custom properties)
export const brand = {
  blue: "#0B4DC7",       // --blue
  blueLight: "#235FCD",  // --blue-2
  blueDark: "#0A45B2",   // --blue-3
  purple: "#333C75",     // --purple
  greyLight: "#F5F5F5",  // --grey-light
  greyLightV2: "#F0F0F0",
  greyDark: "#727B80",   // --grey-dark
  black: "#000000",
  white: "#FFFFFF",
} as const;

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: brand.blue, light: brand.blueLight, dark: brand.blueDark },
    secondary: { main: brand.purple },
    background: { default: brand.greyLight, paper: brand.white },
    text: { primary: brand.black, secondary: brand.greyDark },
  },
  typography: {
    fontFamily: '"Space Grotesk", "Helvetica Neue", Arial, sans-serif',
    h1: { fontWeight: 700, letterSpacing: "-0.03em" },
    h2: { fontWeight: 700, letterSpacing: "-0.02em" },
    h3: { fontWeight: 600, letterSpacing: "-0.02em" },
    h4: { fontWeight: 600, letterSpacing: "-0.01em" },
    button: { textTransform: "none", fontWeight: 600 },
  },
  shape: { borderRadius: 10 },
  components: {
    MuiAccordion: {
      styleOverrides: {
        root: {
          backgroundColor: brand.white,
          borderRadius: 10,
          boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
          border: `1px solid ${brand.greyLightV2}`,
          marginBottom: 8,
          "&:before": { display: "none" },
          "&.Mui-expanded": {
            marginBottom: 8,
            boxShadow: "0 6px 18px rgba(11, 77, 199, 0.10)",
            borderColor: brand.blueLight,
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 999,
          fontWeight: 500,
          fontSize: "0.78rem",
        },
      },
    },
  },
});
