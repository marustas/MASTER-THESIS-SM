export interface Job {
  rank: number;
  job_title: string;
  employer_sector: string;
  location: string;
  job_url: string;
  hybrid_score: number;
  semantic_score: number;
  top_skill_gaps: string[];
}

export interface Programme {
  programme_name: string;
  institution: string;
  jobs: Job[];
}
