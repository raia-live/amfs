/** Portable hosted decision API. Recommendations never execute tools. */
export interface VerificationCheck {
  id: string; instructions: string; observation_tool: string; window_seconds: number;
}
export interface DecisionCandidate {
  id: string; description: string; kind?: "action" | "observation" | "defer";
  cost?: number; verification?: VerificationCheck[];
}
export interface DecisionQuestion {
  type?: "choice" | "boolean" | "score";
  instructions: string; candidates: DecisionCandidate[];
  depends_on?: string[]; levels?: number[];
}
export interface DecideRequest {
  schema_version?: "decision.v1"; decision: string; spec_version: string;
  state: unknown; questions: Record<string, DecisionQuestion>;
  model?: string; mode?: "observe" | "route"; max_error_rate?: number;
  allowed_candidates?: Record<string, string[]>;
  valid_tuples?: Record<string, string>[]; history?: Record<string, unknown>[];
}
export interface DecisionAnswer {
  value: string | boolean | number | null; candidate_id: string | null;
  distribution: Record<string, number>; decision_probability: number | null;
  estimated_success: number | null; automate: boolean;
  disposition: "act" | "gather" | "defer"; reason: string;
  served_by: string; model_version: string;
  verification: VerificationCheck[];
  candidate_distribution: Record<string, number> | null;
  risk: { calibration_id: string; basis: "adjudicated_decision_error";
    sample_size: number; errors: number; upper_error_bound: number;
    confidence_level: number; expires_at: string } | null;
}
export interface DecideResponse {
  schema_version: "decision.v1"; decision_id: string; spec_hash: string;
  answers: Record<string, DecisionAnswer>; automate: boolean;
  capture_status: "durable"; content_hash: string; answered_units: number; created_at: string;
}
export interface DecisionOutcome {
  event_id: string; question: string; candidate_id: string;
  execution: "executed" | "overridden" | "not_executed";
  outcome: "success" | "failure" | "unknown";
  observed_at: string; source: string; evidence_ref?: string;
  verification?: "self_reported" | "external"; supersedes?: string;
}

export interface DecisionModelSpec {
  decision: string; spec_version: string; questions: Record<string, DecisionQuestion>;
  valid_tuples?: Record<string, string>[];
}
export interface DecisionModel {
  id: string; name: string; spec: DecisionModelSpec; spec_hash: string;
  serving_mode: "observe" | "route"; active_version: string | null; created_at: string;
}

export interface DecisionVersion {
  version: string; artifact_sha256: string; evaluation: Record<string, unknown>;
  calibration: Record<string, unknown>; created_at: string;
}
export interface DecisionDataset { id: string; name: string; manifest_sha256: string }
export interface DecisionTrainingJob {
  id: string; model_id: string; version: string; dataset_id: string;
  state: "queued" | "submitting" | "running" | "succeeded" | "failed";
  attempts: number; error_code: string | null; created_at: string; updated_at: string;
}
export interface DecisionUsage {
  decision_count: number; answered_units: number; automated_decisions: number;
  automation_rate: number | null; days: number; since: string;
  by_engine: { engine: string; question_count: number }[];
  daily: { date: string; decision_count: number; answered_units: number; automated_decisions: number }[];
  billing_status: "preview_unbilled"; cost_usd: null;
}
export interface DecisionSummary {
  decision_id: string; created_at: string; answered_units: number; automate: boolean;
  answers: Record<string, Pick<DecisionAnswer, "candidate_id" | "served_by" | "model_version" | "reason" | "disposition" | "automate">>;
}
export interface DecisionHistory { decisions: DecisionSummary[]; next_cursor: string | null; usage: DecisionUsage }
export interface DecisionDetail {
  request: DecideRequest; response: DecideResponse;
  outcomes: { event: DecisionOutcome; verified: boolean }[];
  integrity: { status: "verified" | "invalid" | "key_unavailable"; content_valid: boolean;
    signature_valid: boolean | null; signing_key_id: string };
}

export class DecisionClient {
  constructor(private options: { baseUrl: string; apiKey: string; fetch?: typeof fetch; timeoutMs?: number }) {}

  private async request<T>(path: string, body?: unknown, key?: string, resource = "decisions", method?: "GET" | "POST" | "PATCH"): Promise<T> {
    const response = await (this.options.fetch ?? fetch)(
      `${this.options.baseUrl.replace(/\/$/, "")}/api/v1/${resource}${path}`, {
        method: method ?? (body === undefined ? "GET" : "POST"),
        headers: { "Content-Type": "application/json", "X-AMFS-API-Key": this.options.apiKey,
          ...(key ? { "Idempotency-Key": key } : {}) },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(this.options.timeoutMs ?? 60_000),
      });
    if (!response.ok) throw new Error(`Decision API returned HTTP ${response.status}`);
    return await response.json() as T;
  }

  decide(body: DecideRequest, idempotencyKey = crypto.randomUUID()): Promise<DecideResponse> {
    return this.request("", body, idempotencyKey);
  }
  createModel(name: string, spec: DecisionModelSpec): Promise<DecisionModel> {
    return this.request("", { name, spec }, undefined, "decision-models");
  }
  async listModels(): Promise<DecisionModel[]> {
    const response = await this.request<{ models: DecisionModel[] }>("", undefined, undefined, "decision-models");
    return response.models;
  }
  getModel(name: string): Promise<DecisionModel> {
    return this.request(`/${encodeURIComponent(name)}`, undefined, undefined, "decision-models");
  }
  setMode(name: string, servingMode: "observe" | "route"): Promise<DecisionModel> {
    return this.request(`/${encodeURIComponent(name)}`, { serving_mode: servingMode }, undefined, "decision-models", "PATCH");
  }
  async listVersions(name: string): Promise<DecisionVersion[]> {
    return (await this.request<{ versions: DecisionVersion[] }>(`/${encodeURIComponent(name)}/versions`, undefined, undefined, "decision-models")).versions;
  }
  async listDatasets(name: string): Promise<DecisionDataset[]> {
    return (await this.request<{ datasets: DecisionDataset[] }>(`/${encodeURIComponent(name)}/datasets`, undefined, undefined, "decision-models")).datasets;
  }
  async listJobs(name: string): Promise<DecisionTrainingJob[]> {
    return (await this.request<{ jobs: DecisionTrainingJob[] }>(`/${encodeURIComponent(name)}/jobs`, undefined, undefined, "decision-models")).jobs;
  }
  /** Reuse this explicit key and payload to retry an uncertain response. */
  enqueueTraining(name: string, body: { version: string; dataset_id: string }, idempotencyKey: string): Promise<DecisionTrainingJob> {
    if (!idempotencyKey || idempotencyKey.length > 128) throw new Error("idempotencyKey must contain 1 to 128 characters");
    return this.request(`/${encodeURIComponent(name)}/jobs`, body, idempotencyKey, "decision-models");
  }
  /** Compare-and-set activation. null means first activation; every activation starts in observe mode. */
  activateVersion(name: string, version: string, expectedActiveVersion: string | null): Promise<DecisionModel> {
    if (expectedActiveVersion === undefined) throw new Error("expectedActiveVersion is required, including null for first activation");
    return this.request(`/${encodeURIComponent(name)}/versions/${encodeURIComponent(version)}/activate`,
      { expected_active_version: expectedActiveVersion }, undefined, "decision-models");
  }
  history(name: string, options: { limit?: number; cursor?: string } = {}): Promise<DecisionHistory> {
    const limit = options.limit ?? 50;
    if (!Number.isInteger(limit) || limit < 1 || limit > 100) throw new Error("limit must be between 1 and 100");
    const query = new URLSearchParams({ limit: String(limit) });
    if (options.cursor !== undefined) query.set("cursor", options.cursor);
    return this.request(`/${encodeURIComponent(name)}/decisions?${query}`, undefined, undefined, "decision-models");
  }
  usage(name: string, days = 30): Promise<DecisionUsage> {
    if (!Number.isInteger(days) || days < 1 || days > 90) throw new Error("days must be between 1 and 90");
    return this.request(`/${encodeURIComponent(name)}/usage?days=${days}`, undefined, undefined, "decision-models");
  }
  reportOutcome(decisionId: string, event: DecisionOutcome): Promise<{ event_id: string; verified: boolean }> {
    return this.request(`/${encodeURIComponent(decisionId)}/outcomes`, event);
  }
  get(decisionId: string): Promise<DecisionDetail> {
    return this.request(`/${encodeURIComponent(decisionId)}`);
  }
}
