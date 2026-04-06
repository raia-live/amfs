const BASE = process.env.NEXT_PUBLIC_AMFS_API_URL || "/api/v1";

function headers(): HeadersInit {
  const h: HeadersInit = { "Content-Type": "application/json" };
  if (typeof window !== "undefined") {
    const key = localStorage.getItem("amfs_api_key");
    if (key) h["X-AMFS-API-Key"] = key;
    const branch = localStorage.getItem("amfs_branch");
    if (branch) h["X-AMFS-Branch"] = branch;
  }
  return h;
}

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: headers(), ...opts });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export interface TimelineEvent {
  id: string;
  agent_id: string;
  branch: string;
  event_type: string;
  summary: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface Branch {
  id: string;
  name: string;
  parent_branch: string;
  status: string;
  description: string | null;
  created_at: string;
  entry_count: number;
}

export interface DiffEntry {
  entity_path: string;
  key: string;
  change_type: string;
  branch_value: unknown;
  main_value: unknown;
}

export interface PullRequest {
  id: string;
  title: string;
  description: string | null;
  source_branch: string;
  target_branch: string;
  status: string;
  created_by: string;
  created_at: string;
  reviews?: PRReview[];
}

export interface PRReview {
  id: string;
  pr_id: string;
  reviewer: string;
  status: string;
  comment: string | null;
  created_at: string;
}

export interface Tag {
  id: string;
  name: string;
  branch: string;
  description: string | null;
  tagged_at: string;
}

export interface MergeResult {
  merged: number;
  conflicts: MergeConflict[];
}

export interface MergeConflict {
  entity_path: string;
  key: string;
  branch_value: unknown;
  main_value: unknown;
}

export const api = {
  timeline(agentId: string, opts?: { branch?: string; limit?: number }) {
    const p = new URLSearchParams();
    if (opts?.branch) p.set("branch", opts.branch);
    if (opts?.limit) p.set("limit", String(opts.limit));
    return request<{ events: TimelineEvent[] }>(
      `/agents/${agentId}/timeline?${p}`,
    );
  },

  listBranches() {
    return request<{ branches: Branch[] }>("/branches");
  },

  createBranch(name: string, parentBranch = "main", description?: string) {
    return request<Branch>("/branches", {
      method: "POST",
      body: JSON.stringify({ name, parent_branch: parentBranch, description }),
    });
  },

  getBranch(name: string) {
    return request<Branch>(`/branches/${name}`);
  },

  closeBranch(name: string) {
    return request<{ closed: boolean }>(`/branches/${name}`, {
      method: "DELETE",
    });
  },

  diffBranch(name: string) {
    return request<{ diff: DiffEntry[] }>(`/branches/${name}/diff`);
  },

  mergeBranch(name: string, strategy = "fast_forward") {
    return request<MergeResult>(`/branches/${name}/merge`, {
      method: "POST",
      body: JSON.stringify({ strategy }),
    });
  },

  listPullRequests(status?: string) {
    const p = status ? `?status=${status}` : "";
    return request<{ pull_requests: PullRequest[] }>(`/pull-requests${p}`);
  },

  createPullRequest(
    title: string,
    sourceBranch: string,
    targetBranch = "main",
    description?: string,
  ) {
    const p = new URLSearchParams({ title, source_branch: sourceBranch, target_branch: targetBranch });
    if (description) p.set("description", description);
    return request<PullRequest>(`/pull-requests?${p}`, { method: "POST" });
  },

  getPullRequest(id: string) {
    return request<PullRequest & { reviews: PRReview[] }>(`/pull-requests/${id}`);
  },

  reviewPullRequest(id: string, status: string, comment?: string) {
    const p = new URLSearchParams({ status });
    if (comment) p.set("comment", comment);
    return request<PRReview>(`/pull-requests/${id}/reviews?${p}`, { method: "POST" });
  },

  mergePullRequest(id: string, strategy = "fast_forward") {
    return request<MergeResult>(`/pull-requests/${id}/merge?strategy=${strategy}`, {
      method: "POST",
    });
  },

  closePullRequest(id: string) {
    return request<{ closed: boolean }>(`/pull-requests/${id}/close`, {
      method: "POST",
    });
  },

  listTags() {
    return request<{ tags: Tag[] }>("/tags");
  },

  createTag(name: string, branch = "main", description?: string) {
    return request<Tag>("/tags", {
      method: "POST",
      body: JSON.stringify({ name, branch, description }),
    });
  },

  deleteTag(name: string) {
    return request<{ deleted: boolean }>(`/tags/${name}`, { method: "DELETE" });
  },

  rollback(timestamp?: string, eventId?: string) {
    return request<{ entries_restored: number }>("/rollback", {
      method: "POST",
      body: JSON.stringify({
        target_timestamp: timestamp,
        target_event_id: eventId,
      }),
    });
  },

  rollbackToTag(tagName: string) {
    return request<{ entries_restored: number }>(`/rollback/tag/${tagName}`, {
      method: "POST",
    });
  },

  fork(targetAgentId: string) {
    return request<{ entries_copied: number }>(
      `/fork?target_agent_id=${targetAgentId}`,
      { method: "POST" },
    );
  },
};
