import { useMemo, useState } from "react";
import {
  Autocomplete,
  Box,
  Container,
  CssBaseline,
  ThemeProvider,
  TextField,
  Typography,
} from "@mui/material";
import { theme, brand } from "./theme";
import rankings from "./data/rankings.json";
import type { Programme } from "./types";
import { JobAccordion } from "./components/JobAccordion";

const programmes = rankings as Programme[];

export default function App() {
  const [selected, setSelected] = useState<Programme | null>(null);

  const options = useMemo(
    () =>
      [...programmes].sort((a, b) =>
        a.programme_name.localeCompare(b.programme_name),
      ),
    [],
  );

  const corpusMaxScore = useMemo(
    () =>
      Math.max(
        ...programmes.flatMap((p) => p.jobs.map((j) => j.hybrid_score)),
      ),
    [],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
      <Box sx={{ bgcolor: brand.greyLight, minHeight: "100vh", py: { xs: 4, md: 6 } }}>
        <Container maxWidth="md">
          <Box sx={{ borderLeft: `4px solid ${brand.blue}`, pl: 2, mb: 4 }}>
            <Typography
              variant="overline"
              sx={{ color: brand.greyDark, letterSpacing: "0.12em", display: "block" }}
            >
              Master Thesis · Vilnius Tech
            </Typography>
            <Typography variant="h4" sx={{ color: brand.blueDark, mt: 0.5 }}>
              Study Programme → Job Market Alignment
            </Typography>
          </Box>

          <Autocomplete
            options={options}
            getOptionLabel={(o) => `${o.programme_name} — ${o.institution}`}
            value={selected}
            onChange={(_, v) => setSelected(v)}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Choose a study programme"
                placeholder="Start typing or pick from the list…"
              />
            )}
            sx={{ mb: 4 }}
          />

          {selected ? (
            <>
              <Box
                sx={{
                  bgcolor: brand.blueDark,
                  color: brand.white,
                  borderRadius: 1.5,
                  px: 2.5,
                  py: 1.75,
                  mb: 3,
                  animation: "fadeUp 0.4s ease-out both",
                }}
              >
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  {selected.programme_name}
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                  {selected.institution}
                </Typography>
              </Box>

              {selected.jobs.map((job, i) => (
                <JobAccordion
                  key={`${selected.programme_name}-${job.rank}`}
                  job={job}
                  corpusMaxScore={corpusMaxScore}
                  delayMs={60 * i}
                />
              ))}
            </>
          ) : (
            <Typography sx={{ color: brand.greyDark }}>
              Pick a programme above to see the top-10 hybrid matches from the job-ad corpus.
            </Typography>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
