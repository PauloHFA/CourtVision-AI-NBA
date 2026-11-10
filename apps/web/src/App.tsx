import { useEffect, useState, type FormEvent } from "react";
import {
  ask,
  getGames,
  getHealth,
  getLatestEvaluation,
  getOverview,
  getRagEvaluation,
  getTeams,
  predict,
  type AskResponse,
  type GameRow,
  type LatestEvaluation,
  type Overview,
  type Prediction,
  type RagReport,
  type Team,
} from "./api";

type Tab = "painel" | "comparador" | "assistente" | "laboratorio" | "sobre";

const DEFAULT_AS_OF = "2024-04-01T18:00";
const DEFAULT_QUESTION = "Quais jogos anteriores sustentam a análise?";

export function App() {
  const [tab, setTab] = useState<Tab>("comparador");
  const [teams, setTeams] = useState<Team[]>([]);
  const [health, setHealth] = useState<string>("Consultando a API local…");
  const [error, setError] = useState<string>("");
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [asOf, setAsOf] = useState(DEFAULT_AS_OF);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);

  useEffect(() => {
    getHealth()
      .then((body) => {
        setHealth(`API ${body.status}. Dados: ${body.components.database}. Modelo: ${body.components.model}.`);
      })
      .catch(() => setHealth("API indisponível em http://127.0.0.1:8000."));
    getTeams()
      .then((body) => setTeams(body.data))
      .catch((reason: Error) => setError(reason.message));
  }, []);

  function openEvidence() {
    setQuestion(DEFAULT_QUESTION);
    setTab("assistente");
  }

  return (
    <div className="wrap">
      <header>
        <h1>CourtVision AI</h1>
        <p className="disclaimer">
          Estimativa estatística educacional. Não é afiliado à NBA, à BALLDONTLIE ou a equipes.
          Não recomenda apostas e não garante resultado.
        </p>
        <p className="note">{health}</p>
      </header>
      <nav aria-label="Seções">
        {(
          [
            ["painel", "Painel"],
            ["comparador", "Comparador"],
            ["assistente", "Assistente"],
            ["laboratorio", "Laboratório"],
            ["sobre", "Sobre os dados"],
          ] as const
        ).map(([id, label]) => (
          <button key={id} type="button" aria-selected={tab === id} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </nav>
      {error ? <p className="error" role="alert">{error}</p> : null}
      {tab === "painel" ? <Panel /> : null}
      {tab === "comparador" ? (
        <Comparator
          teams={teams}
          home={home}
          away={away}
          asOf={asOf}
          onHome={setHome}
          onAway={setAway}
          onAsOf={setAsOf}
          onEvidence={openEvidence}
        />
      ) : null}
      {tab === "assistente" ? (
        <Assistant
          teams={teams}
          home={home}
          away={away}
          asOf={asOf}
          question={question}
          onHome={setHome}
          onAway={setAway}
          onAsOf={setAsOf}
          onQuestion={setQuestion}
        />
      ) : null}
      {tab === "laboratorio" ? <Lab /> : null}
      {tab === "sobre" ? <About /> : null}
    </div>
  );
}

function Panel() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [season, setSeason] = useState<number | null>(null);
  const [games, setGames] = useState<GameRow[]>([]);
  const [teams, setTeams] = useState<Team[]>([]);

  useEffect(() => {
    getOverview()
      .then((body) => {
        setOverview(body);
        setSeason(body.seasons.at(-1)?.season ?? null);
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
    getTeams()
      .then((body) => setTeams(body.data))
      .catch(() => setTeams([]));
  }, []);

  useEffect(() => {
    if (season == null) return;
    getGames({ season, limit: 8 })
      .then((body) => setGames(body.data))
      .catch(() => setGames([]));
  }, [season]);

  return (
    <section className="panel" aria-labelledby="painel-titulo">
      <h2 id="painel-titulo">Painel</h2>
      {loading ? <p>Carregando visão geral…</p> : null}
      {error ? <p className="error" role="alert">{error}</p> : null}
      {overview ? (
        <>
          <p className="note">
            {overview.data_source === "fixtures"
              ? "Fonte local: fixture sintético, não um histórico oficial."
              : `Fonte: ${overview.data_source}.`}
            {" "}A avaliação externa não está verificada.
          </p>
          <div className="stats">
            <Stat label="Equipes" value={String(overview.teams)} />
            <Stat label="Jogos" value={String(overview.games)} />
            <Stat label="Modelo em execução" value={overview.runtime_model} />
            <Stat label="Versão" value={overview.model_version} />
          </div>
          <p className="note">Features {overview.feature_version}. Hash do conjunto: {overview.dataset_hash.slice(0, 12)}.</p>
          <h3>Temporadas disponíveis</h3>
          <table>
            <thead>
              <tr>
                <th>Temporada</th>
                <th>Jogos</th>
                <th>Finalizados</th>
                <th>Agendados</th>
              </tr>
            </thead>
            <tbody>
              {overview.seasons.map((row) => (
                <tr key={row.season}>
                  <td>
                    <button type="button" className="link" onClick={() => setSeason(row.season)}>
                      {row.season}
                    </button>
                  </td>
                  <td>{row.games}</td>
                  <td>{row.final}</td>
                  <td>{row.scheduled}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {season != null ? (
            <>
              <h3>Amostra retrospectiva de {season}</h3>
              <GameList games={games} teams={teams} />
            </>
          ) : null}
          <p className="note">{overview.disclaimer}</p>
        </>
      ) : null}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Comparator({
  teams,
  home,
  away,
  asOf,
  onHome,
  onAway,
  onAsOf,
  onEvidence,
}: {
  teams: Team[];
  home: string;
  away: string;
  asOf: string;
  onHome: (value: string) => void;
  onAway: (value: string) => void;
  onAsOf: (value: string) => void;
  onEvidence: () => void;
}) {
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setResult(null);
    if (home === away) {
      setError("Mandante e visitante devem ser equipes diferentes.");
      return;
    }
    setLoading(true);
    try {
      setResult(await predict(Number(home), Number(away), new Date(asOf).toISOString()));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Falha ao prever.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="comparador-titulo">
      <h2 id="comparador-titulo">Comparador pré-jogo</h2>
      {teams.length === 0 ? <p>Nenhuma equipe carregada.</p> : null}
      <form onSubmit={onSubmit}>
        <div className="row">
          <TeamField id="home" label="Mandante" teams={teams} value={home} onChange={onHome} />
          <TeamField id="away" label="Visitante" teams={teams} value={away} onChange={onAway} />
        </div>
        <label htmlFor="asof">
          Data de corte
          <input id="asof" type="datetime-local" value={asOf} onChange={(event) => onAsOf(event.target.value)} required />
        </label>
        <button className="primary" type="submit" disabled={loading || teams.length === 0}>
          {loading ? "Calculando…" : "Prever"}
        </button>
      </form>
      {error ? <p className="error" role="alert">{error}</p> : null}
      {result ? <PredictionView result={result} onEvidence={onEvidence} /> : null}
    </section>
  );
}

function TeamField({
  id,
  label,
  teams,
  value,
  onChange,
}: {
  id: string;
  label: string;
  teams: Team[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label htmlFor={id}>
      {label}
      <select id={id} value={value} onChange={(event) => onChange(event.target.value)} required>
        <option value="">Selecione</option>
        {teams.map((team) => (
          <option key={team.id} value={team.id}>
            {team.name} ({team.abbreviation})
          </option>
        ))}
      </select>
    </label>
  );
}

function PredictionView({ result, onEvidence }: { result: Prediction; onEvidence: () => void }) {
  if (result.abstained || result.p_home_win === null || result.p_away_win === null) {
    return (
      <div aria-live="polite">
        <p className="error">{result.abstain_reason ?? "Sem estimativa operacional."}</p>
        <p className="note">Cobertura: {result.data_quality}. Sem percentual inventado.</p>
        <p className="note">{result.disclaimer}</p>
      </div>
    );
  }
  const home = Math.round(result.p_home_win * 1000) / 10;
  const away = Math.round(result.p_away_win * 1000) / 10;
  return (
    <div aria-live="polite">
      <p>Estimativa estatística pré-jogo. Modelo {result.model_version}. Features {result.feature_version}.</p>
      <p className="note">Cobertura: {result.data_quality}. Soma exibida: {(home + away).toFixed(1)}% (arredondamento de 1 casa).</p>
      <div className="bar">
        <span>Mandante: {home.toFixed(1)}%</span>
        <div className="track" aria-hidden="true">
          <div className="fill" style={{ width: `${home}%` }} />
        </div>
        <span>Visitante: {away.toFixed(1)}%</span>
      </div>
      <ul>
        {result.factors.map((factor) => (
          <li key={factor.feature}>
            {factor.label}: {factor.value ?? "—"}
          </li>
        ))}
      </ul>
      {result.feature_hash ? <p className="note">Hash das features: {result.feature_hash.slice(0, 12)}.</p> : null}
      <button type="button" onClick={onEvidence}>Ver evidências</button>
      <p className="note">{result.disclaimer}</p>
    </div>
  );
}

function Assistant({
  teams,
  home,
  away,
  asOf,
  question,
  onHome,
  onAway,
  onAsOf,
  onQuestion,
}: {
  teams: Team[];
  home: string;
  away: string;
  asOf: string;
  question: string;
  onHome: (value: string) => void;
  onAway: (value: string) => void;
  onAsOf: (value: string) => void;
  onQuestion: (value: string) => void;
}) {
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [evidence, setEvidence] = useState<GameRow | null>(null);
  const [evidenceNote, setEvidenceNote] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setAnswer(null);
    setEvidence(null);
    setEvidenceNote("");
    if (home === away) {
      setError("Mandante e visitante devem ser equipes diferentes.");
      return;
    }
    setLoading(true);
    try {
      setAnswer(await ask(question, new Date(asOf).toISOString(), Number(home), Number(away)));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Falha na consulta.");
    } finally {
      setLoading(false);
    }
  }

  async function openCitation(sourceId: string) {
    setEvidence(null);
    setEvidenceNote("");
    const match = /^game:(\d+)$/.exec(sourceId);
    if (!match) {
      setEvidenceNote(`${sourceId} não é um jogo consultável.`);
      return;
    }
    try {
      const body = await getGames({ sourceGameId: Number(match[1]), limit: 1 });
      const game = body.data[0];
      if (!game) {
        setEvidenceNote("Jogo citado não encontrado na fonte local.");
        return;
      }
      setEvidence(game);
    } catch (reason) {
      setEvidenceNote(reason instanceof Error ? reason.message : "Falha ao abrir a evidência.");
    }
  }

  return (
    <section className="panel" aria-labelledby="assistente-titulo">
      <h2 id="assistente-titulo">Assistente</h2>
      <p className="note">A resposta copia a probabilidade do modelo e cita evidências. Não inventa percentual.</p>
      <form onSubmit={onSubmit}>
        <div className="row">
          <TeamField id="ask-home" label="Mandante" teams={teams} value={home} onChange={onHome} />
          <TeamField id="ask-away" label="Visitante" teams={teams} value={away} onChange={onAway} />
        </div>
        <label htmlFor="ask-asof">
          Data de corte
          <input id="ask-asof" type="datetime-local" value={asOf} onChange={(event) => onAsOf(event.target.value)} required />
        </label>
        <label htmlFor="question">
          Pergunta
          <textarea id="question" value={question} onChange={(event) => onQuestion(event.target.value)} required rows={3} />
        </label>
        <button className="primary" type="submit" disabled={loading}>
          {loading ? "Consultando…" : "Perguntar"}
        </button>
      </form>
      {error ? <p className="error" role="alert">{error}</p> : null}
      {answer ? (
        <div aria-live="polite">
          <p>{answer.answer}</p>
          {answer.insufficient_evidence ? <p className="note">Evidência insuficiente para ir além do que foi citado.</p> : null}
          {answer.p_home_win === null ? <p className="note">Sem percentual do modelo nesta resposta.</p> : null}
          {answer.facts.length > 0 ? (
            <ul>
              {answer.facts.map((fact) => (
                <li key={fact.source_ids.join("-")}>
                  {fact.metric}: {fact.value} em {fact.sample} jogos anteriores.
                </li>
              ))}
            </ul>
          ) : null}
          <ul>
            {answer.citations.map((citation) => (
              <li key={citation.source_id}>
                <button type="button" className="link" onClick={() => openCitation(citation.source_id)}>
                  {citation.source_id}
                </button>
                {" "}({citation.source_type})
              </li>
            ))}
          </ul>
          {evidenceNote ? <p className="note">{evidenceNote}</p> : null}
          {evidence ? <EvidenceCard game={evidence} teams={teams} cutoff={asOf} /> : null}
        </div>
      ) : null}
    </section>
  );
}

function EvidenceCard({ game, teams, cutoff }: { game: GameRow; teams: Team[]; cutoff: string }) {
  const cutoffDay = cutoff.slice(0, 10);
  const tooLate = game.game_date >= cutoffDay;
  return (
    <div className="evidence">
      <p>
        {teamName(teams, game.home_team_id)} x {teamName(teams, game.visitor_team_id)} em {game.game_date}.
      </p>
      {tooLate ? (
        <p className="error">Este jogo não é anterior ao corte e não entra como evidência pré-jogo.</p>
      ) : (
        <p>
          Histórico retrospectivo. Status: {game.status}. Placar: {scoreText(game)}.
        </p>
      )}
    </div>
  );
}

function GameList({ games, teams }: { games: GameRow[]; teams: Team[] }) {
  if (games.length === 0) return <p>Sem jogos nesta amostra.</p>;
  return (
    <ul>
      {games.map((game) => (
        <li key={game.id}>
          {game.game_date}: {teamName(teams, game.home_team_id)} {scoreText(game)} {teamName(teams, game.visitor_team_id)} ({game.status})
        </li>
      ))}
    </ul>
  );
}

function teamName(teams: Team[], id: number) {
  return teams.find((team) => team.id === id)?.name ?? `Equipe ${id}`;
}

function scoreText(game: GameRow) {
  if (game.home_score === null || game.visitor_score === null) return "sem placar";
  return `${game.home_score} a ${game.visitor_score}`;
}

function Lab() {
  const [latest, setLatest] = useState<LatestEvaluation | null>(null);
  const [report, setReport] = useState<RagReport | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getLatestEvaluation()
      .then(setLatest)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  async function runProtocol() {
    setError("");
    setLoading(true);
    try {
      setReport(await getRagEvaluation());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Falha ao rodar o protocolo.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="lab-titulo">
      <h2 id="lab-titulo">Laboratório</h2>
      <p className="note">Não há gráfico de avaliação externa. Metas do documento não são resultados medidos.</p>
      {latest ? (
        <p>{latest.message} Desempenho verificado: {latest.performance_verified ? "sim" : "não"}.</p>
      ) : (
        <p>Consultando o último relatório…</p>
      )}
      <button type="button" className="primary" onClick={runProtocol} disabled={loading}>
        {loading ? "Rodando protocolo sintético…" : "Rodar protocolo sintético"}
      </button>
      {error ? <p className="error" role="alert">{error}</p> : null}
      {report ? (
        <div aria-live="polite">
          <p>{report.note}</p>
          <p>
            Perguntas: {report.question_count}. Aceite AC-08: {report.ac08_accepted ? "sim" : "não"}.
            Desempenho verificado: {report.performance_verified ? "sim" : "não"}.
          </p>
          <ul>
            {Object.entries(report.metrics).map(([name, value]) => (
              <li key={name}>{name}: {value === null ? "—" : value}</li>
            ))}
          </ul>
          {report.failures.length > 0 ? (
            <>
              <h3>Falhas desta rodada</h3>
              <ul>
                {report.failures.slice(0, 5).map((failure) => (
                  <li key={failure.id}>{failure.id} ({failure.category}): {failure.reasons.join(", ")}</li>
                ))}
              </ul>
            </>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function About() {
  return (
    <section className="panel" aria-labelledby="sobre-titulo">
      <h2 id="sobre-titulo">Sobre os dados</h2>
      <ul>
        <li>Fonte prevista: BALLDONTLIE, rotas de equipes e jogos do plano gratuito.</li>
        <li>O repositório público traz apenas fixtures sintéticos.</li>
        <li>Termos conferidos no documento em 08/11/2026. Confira de novo antes de publicar dados.</li>
        <li>Não usamos logos, fotos ou scraping de sites oficiais.</li>
        <li>Simulação de jogador ausente continua fora do MVP.</li>
      </ul>
    </section>
  );
}
