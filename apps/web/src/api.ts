const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export type Team = {
  id: number;
  name: string;
  abbreviation: string;
  city: string | null;
};

export type Prediction = {
  prediction_id: string | null;
  p_home_win: number | null;
  p_away_win: number | null;
  model_version: string;
  feature_version: string;
  as_of: string;
  data_quality: string;
  abstained: boolean;
  abstain_reason: string | null;
  factors: { feature: string; label: string; value: number | null }[];
  disclaimer: string;
  feature_hash?: string;
  dataset_hash?: string | null;
};

export type Citation = {
  source_id: string;
  source_type: string;
  available_at?: string;
  as_of?: string;
};

export type AskResponse = {
  answer: string;
  facts: { metric: string; value: number; sample: number; source_ids: string[] }[];
  citations: Citation[];
  insufficient_evidence: boolean;
  p_home_win: number | null;
  model_version: string | null;
};

export type Overview = {
  teams: number;
  games: number;
  seasons: { season: number; games: number; final: number; scheduled: number }[];
  model_version: string;
  runtime_model: string;
  feature_version: string;
  dataset_hash: string;
  data_source: string;
  performance_verified: boolean;
  disclaimer: string;
};

export type LatestEvaluation = {
  available: boolean;
  performance_verified: boolean;
  message: string;
  rag_protocol: { path: string; acceptance_set: boolean; note: string };
};

export type RagReport = {
  dataset: string;
  condition: string;
  acceptance_set: boolean;
  ac08_accepted: boolean;
  performance_verified: boolean;
  question_count: number;
  by_category: Record<string, number>;
  note: string;
  metrics: Record<string, number | null>;
  failures: { id: string; category: string; reasons: string[] }[];
};

export type GameRow = {
  id: number;
  source_game_id: number | null;
  season: number;
  game_date: string;
  status: string;
  home_team_id: number;
  visitor_team_id: number;
  home_score: number | null;
  visitor_score: number | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  const body = await response.json();
  if (!response.ok) {
    const message = body?.detail?.message ?? "A API recusou a consulta.";
    throw new Error(message);
  }
  return body as T;
}

export function getHealth() {
  return request<{ status: string; components: Record<string, string>; disclaimer: string }>("/api/v1/health");
}

export function getTeams() {
  return request<{ data: Team[] }>("/api/v1/teams");
}

export function getOverview() {
  return request<Overview>("/api/v1/overview");
}

export function getLatestEvaluation() {
  return request<LatestEvaluation>("/api/v1/evaluations/latest");
}

export function getRagEvaluation() {
  return request<RagReport>("/api/v1/evaluations/rag");
}

export function getGames(params: { sourceGameId?: number; limit?: number; season?: number } = {}) {
  const search = new URLSearchParams();
  if (params.sourceGameId != null) search.set("source_game_id", String(params.sourceGameId));
  if (params.limit != null) search.set("limit", String(params.limit));
  if (params.season != null) search.set("season", String(params.season));
  const query = search.toString();
  return request<{ data: GameRow[]; meta: { limit: number; offset: number; count: number } }>(
    `/api/v1/games${query ? `?${query}` : ""}`,
  );
}

export function predict(homeTeamId: number, awayTeamId: number, asOf: string) {
  return request<Prediction>("/api/v1/predictions", {
    method: "POST",
    body: JSON.stringify({
      home_team_id: homeTeamId,
      away_team_id: awayTeamId,
      as_of: asOf,
    }),
  });
}

export function ask(question: string, asOf: string, homeTeamId: number, awayTeamId: number) {
  return request<AskResponse>("/api/v1/assistant/ask", {
    method: "POST",
    body: JSON.stringify({
      question,
      as_of: asOf,
      home_team_id: homeTeamId,
      away_team_id: awayTeamId,
    }),
  });
}
