import { useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  Chip,
  LinearProgress,
  Stack,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import type { Job } from "../types";
import { brand } from "../theme";

interface Props {
  job: Job;
  maxScore: number;
  delayMs: number;
}

export function JobAccordion({ job, maxScore, delayMs }: Props) {
  const [expanded, setExpanded] = useState(false);
  const scorePct = maxScore === 0 ? 0 : (job.hybrid_score / maxScore) * 100;
  const meta = [job.employer_sector, job.location].filter(Boolean).join(" · ");

  return (
    <Accordion
      expanded={expanded}
      onChange={(_, isExpanded) => setExpanded(isExpanded)}
      disableGutters
      sx={{
        opacity: 0,
        animation: "fadeUp 0.45s ease-out forwards",
        animationDelay: `${delayMs}ms`,
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon sx={{ color: brand.blue }} />}
        sx={{ minHeight: 64, "& .MuiAccordionSummary-content": { alignItems: "center", gap: 2 } }}
      >
        <Box
          sx={{
            flex: "0 0 44px",
            height: 44,
            borderRadius: 1.25,
            bgcolor: brand.blue,
            color: brand.white,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 700,
            fontSize: "0.95rem",
          }}
        >
          #{job.rank}
        </Box>
        <Typography
          variant="subtitle1"
          sx={{ flex: 1, fontWeight: 600, color: brand.black, lineHeight: 1.3 }}
        >
          {job.job_title}
        </Typography>
      </AccordionSummary>

      <AccordionDetails sx={{ pt: 0 }}>
        {meta && (
          <Typography variant="body2" sx={{ color: brand.greyDark, mb: 1.5 }}>
            {meta}
          </Typography>
        )}

        <Stack sx={{ flexDirection: "row", alignItems: "center", gap: 1.5, mb: 1.5 }}>
          <Box sx={{ flex: 1 }}>
            <LinearProgress
              variant="determinate"
              value={scorePct}
              sx={{
                height: 7,
                borderRadius: 4,
                bgcolor: brand.greyLightV2,
                "& .MuiLinearProgress-bar": {
                  background: `linear-gradient(90deg, ${brand.blue} 0%, ${brand.purple} 100%)`,
                  borderRadius: 4,
                },
              }}
            />
          </Box>
          <Typography
            variant="caption"
            sx={{
              fontWeight: 600,
              color: brand.blue,
              fontVariantNumeric: "tabular-nums",
              minWidth: 48,
              textAlign: "right",
            }}
          >
            {job.hybrid_score.toFixed(3)}
          </Typography>
        </Stack>

        {job.top_skill_gaps.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography
              variant="overline"
              sx={{
                color: brand.greyDark,
                letterSpacing: "0.08em",
                fontSize: "0.7rem",
                display: "block",
                mb: 0.5,
              }}
            >
              Skills the programme lacks
            </Typography>
            <Stack sx={{ flexDirection: "row", flexWrap: "wrap", gap: 0.75 }}>
              {job.top_skill_gaps.map((skill) => (
                <Chip
                  key={skill}
                  label={skill}
                  size="small"
                  sx={{ bgcolor: brand.greyLight, color: brand.blueDark }}
                />
              ))}
            </Stack>
          </Box>
        )}

        <Button
          variant="text"
          endIcon={<OpenInNewIcon />}
          href={job.job_url}
          target="_blank"
          rel="noopener"
          sx={{ color: brand.blue, px: 0, "&:hover": { background: "transparent", textDecoration: "underline" } }}
        >
          View job posting
        </Button>
      </AccordionDetails>
    </Accordion>
  );
}
