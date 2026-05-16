import { useMemo, useState } from "react";
import {
  Alert,
  Autocomplete,
  Box,
  Container,
  CssBaseline,
  ThemeProvider,
  TextField,
  Typography,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import { theme, brand } from "./theme";
import rankings from "./data/rankings.json";
import type { Programme } from "./types";
import { JobAccordion } from "./components/JobAccordion";

const programmes = rankings as Programme[];

// Top-1 displayed score below this is treated as "limited corpus coverage" —
// the algorithm picked the least-bad available match rather than a confident
// fit.  Threshold of 35/100 flags the worst third of programmes (15/45) and
// matches the heuristic identified in the domain-expert review.
const LOW_CONFIDENCE_THRESHOLD = 35;

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

  const top1RankQuality = useMemo(() => {
    if (!selected || selected.jobs.length === 0) return 0;
    const topScore = selected.jobs[0].hybrid_score;
    return corpusMaxScore === 0 ? 0 : (topScore / corpusMaxScore) * 100;
  }, [selected, corpusMaxScore]);

  const isLowConfidence = selected !== null && top1RankQuality < LOW_CONFIDENCE_THRESHOLD;

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

              {isLowConfidence && (
                <Alert
                  severity="info"
                  icon={<InfoOutlinedIcon sx={{ color: brand.purple }} />}
                  sx={{
                    mb: 2.5,
                    bgcolor: "#F0F2F8",
                    color: brand.blueDark,
                    border: `1px solid ${brand.greyLightV2}`,
                    "& .MuiAlert-message": { fontSize: "0.85rem" },
                    animation: "fadeUp 0.4s ease-out both",
                  }}
                >
                  Limited corpus coverage — the best match scores
                  {" "}<strong>{Math.round(top1RankQuality)}/100</strong>.
                  Treat the ranking below as candidates to review, not
                  as a confident recommendation.
                </Alert>
              )}

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
