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

export class DecisionClient {
  constructor(private options: { baseUrl: string; apiKey: string; fetch?: typeof fetch; timeoutMs?: number }) {}

  private async request<T>(path: string, body?: unknown, key?: string, resource = "decisions"): Promise<T> {
    const response = await (this.options.fetch ?? fetch)(
      `${this.options.baseUrl.replace(/\/$/, "")}/api/v1/${resource}${path}`, {
        method: body === undefined ? "GET" : "POST",
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
  reportOutcome(decisionId: string, event: DecisionOutcome): Promise<{ event_id: string; verified: boolean }> {
    return this.request(`/${encodeURIComponent(decisionId)}/outcomes`, event);
  }
  get(decisionId: string): Promise<{ request: DecideRequest; response: DecideResponse;
    outcomes: { event: DecisionOutcome; verified: boolean }[] }> {
    return this.request(`/${encodeURIComponent(decisionId)}`);
  }
}
