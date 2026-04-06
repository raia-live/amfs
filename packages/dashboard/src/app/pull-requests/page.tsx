"use client";

import { useState, useEffect } from "react";
import { GitPullRequest, Plus, CheckCircle, XCircle, MessageSquare, Merge, X } from "lucide-react";
import { api, PullRequest } from "@/lib/api";

const STATUS_STYLES: Record<string, { bg: string; text: string }> = {
  open: { bg: "bg-success/20", text: "text-success" },
  merged: { bg: "bg-branch-purple/20", text: "text-branch-purple" },
  closed: { bg: "bg-danger/20", text: "text-danger" },
};

export default function PullRequestsPage() {
  const [prs, setPrs] = useState<PullRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("");
  const [showCreate, setShowCreate] = useState(false);
  const [selectedPR, setSelectedPR] = useState<(PullRequest & { reviews?: any[] }) | null>(null);

  async function fetchPRs() {
    setLoading(true);
    try {
      const data = await api.listPullRequests(filter || undefined);
      setPrs(data.pull_requests || []);
    } catch {
      setPrs([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchPRs();
  }, [filter]);

  async function handleCreate(title: string, source: string, target: string, desc: string) {
    await api.createPullRequest(title, source, target, desc || undefined);
    setShowCreate(false);
    fetchPRs();
  }

  async function handleSelect(pr: PullRequest) {
    try {
      const full = await api.getPullRequest(pr.id);
      setSelectedPR(full);
    } catch {
      setSelectedPR({ ...pr, reviews: [] });
    }
  }

  async function handleReview(status: string, comment?: string) {
    if (!selectedPR) return;
    await api.reviewPullRequest(selectedPR.id, status, comment);
    handleSelect(selectedPR);
  }

  async function handleMerge() {
    if (!selectedPR) return;
    try {
      await api.mergePullRequest(selectedPR.id);
      setSelectedPR(null);
      fetchPRs();
    } catch (e) {
      alert(e instanceof Error ? e.message : "Merge failed");
    }
  }

  async function handleClose() {
    if (!selectedPR) return;
    await api.closePullRequest(selectedPR.id);
    setSelectedPR(null);
    fetchPRs();
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <GitPullRequest className="w-5 h-5 text-branch-purple" />
          <h1 className="text-xl font-semibold">Pull Requests</h1>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="bg-void-gray border border-void-light rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none"
          >
            <option value="">All</option>
            <option value="open">Open</option>
            <option value="merged">Merged</option>
            <option value="closed">Closed</option>
          </select>
          <button
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-1.5 px-3 py-2 bg-branch-purple/20 text-branch-purple border border-branch-purple/30 rounded-lg text-sm hover:bg-branch-purple/30 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New PR
          </button>
        </div>
      </div>

      {showCreate && (
        <CreatePRModal
          onClose={() => setShowCreate(false)}
          onCreate={handleCreate}
        />
      )}

      {loading ? (
        <div className="text-center py-12 text-text-muted">Loading…</div>
      ) : prs.length === 0 ? (
        <div className="text-center py-12 text-text-muted">No pull requests found.</div>
      ) : (
        <div className="space-y-3">
          {prs.map((pr) => {
            const st = STATUS_STYLES[pr.status] || STATUS_STYLES.open;
            return (
              <button
                key={pr.id}
                onClick={() => handleSelect(pr)}
                className="w-full text-left bg-void-dark border border-void-light rounded-xl p-4 hover:border-branch-purple/30 transition-colors"
              >
                <div className="flex items-center gap-2 mb-1">
                  <GitPullRequest className="w-4 h-4 text-branch-purple" />
                  <span className="font-medium">{pr.title}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded ${st.bg} ${st.text}`}>
                    {pr.status}
                  </span>
                </div>
                <p className="text-sm text-text-secondary">
                  <span className="text-branch-blue">{pr.source_branch}</span>
                  {" → "}
                  <span className="text-sacred-gold">{pr.target_branch}</span>
                </p>
                {pr.description && (
                  <p className="text-xs text-text-muted mt-1">{pr.description}</p>
                )}
                <p className="text-xs text-text-muted mt-1">
                  by {pr.created_by} · {new Date(pr.created_at).toLocaleDateString()}
                </p>
              </button>
            );
          })}
        </div>
      )}

      {selectedPR && (
        <PRDetailModal
          pr={selectedPR}
          onClose={() => setSelectedPR(null)}
          onReview={handleReview}
          onMerge={handleMerge}
          onClosePR={handleClose}
        />
      )}
    </div>
  );
}

function CreatePRModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (title: string, source: string, target: string, desc: string) => void;
}) {
  const [title, setTitle] = useState("");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("main");
  const [desc, setDesc] = useState("");

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-void-dark border border-void-light rounded-xl p-6 w-full max-w-md">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">New Pull Request</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="space-y-3">
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Title"
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-purple/50"
          />
          <input
            type="text"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="Source branch"
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-purple/50"
          />
          <input
            type="text"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            placeholder="Target branch (default: main)"
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-purple/50"
          />
          <textarea
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="Description (optional)"
            rows={3}
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-purple/50 resize-none"
          />
          <button
            onClick={() => onCreate(title, source, target, desc)}
            disabled={!title || !source}
            className="w-full py-2 bg-branch-purple/20 text-branch-purple border border-branch-purple/30 rounded-lg text-sm font-medium hover:bg-branch-purple/30 disabled:opacity-40 transition-colors"
          >
            Create Pull Request
          </button>
        </div>
      </div>
    </div>
  );
}

function PRDetailModal({
  pr,
  onClose,
  onReview,
  onMerge,
  onClosePR,
}: {
  pr: PullRequest & { reviews?: any[] };
  onClose: () => void;
  onReview: (status: string, comment?: string) => void;
  onMerge: () => void;
  onClosePR: () => void;
}) {
  const [comment, setComment] = useState("");

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-void-dark border border-void-light rounded-xl p-6 w-full max-w-xl max-h-[80vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">{pr.title}</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="mb-4 text-sm text-text-secondary">
          <span className="text-branch-blue">{pr.source_branch}</span>
          {" → "}
          <span className="text-sacred-gold">{pr.target_branch}</span>
          {pr.description && <p className="mt-2 text-text-muted">{pr.description}</p>}
        </div>

        {pr.reviews && pr.reviews.length > 0 && (
          <div className="mb-4 space-y-2">
            <h3 className="text-sm font-medium">Reviews</h3>
            {pr.reviews.map((r: any, i: number) => (
              <div key={i} className="bg-void-gray rounded-lg p-3 border border-void-light text-sm">
                <div className="flex items-center gap-2">
                  {r.status === "approved" ? (
                    <CheckCircle className="w-4 h-4 text-success" />
                  ) : r.status === "changes_requested" ? (
                    <XCircle className="w-4 h-4 text-danger" />
                  ) : (
                    <MessageSquare className="w-4 h-4 text-text-muted" />
                  )}
                  <span className="text-text-primary">{r.reviewer}</span>
                  <span className="text-text-muted">{r.status}</span>
                </div>
                {r.comment && (
                  <p className="text-text-muted mt-1 ml-6">{r.comment}</p>
                )}
              </div>
            ))}
          </div>
        )}

        {pr.status === "open" && (
          <>
            <div className="mb-4">
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="Review comment (optional)"
                rows={2}
                className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none resize-none"
              />
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => { onReview("approved", comment); setComment(""); }}
                  className="flex items-center gap-1 px-3 py-1.5 bg-success/20 text-success border border-success/30 rounded-lg text-xs hover:bg-success/30 transition-colors"
                >
                  <CheckCircle className="w-3.5 h-3.5" /> Approve
                </button>
                <button
                  onClick={() => { onReview("changes_requested", comment); setComment(""); }}
                  className="flex items-center gap-1 px-3 py-1.5 bg-danger/20 text-danger border border-danger/30 rounded-lg text-xs hover:bg-danger/30 transition-colors"
                >
                  <XCircle className="w-3.5 h-3.5" /> Request Changes
                </button>
                <button
                  onClick={() => { onReview("commented", comment); setComment(""); }}
                  className="flex items-center gap-1 px-3 py-1.5 bg-void-gray text-text-secondary border border-void-light rounded-lg text-xs hover:bg-void-light transition-colors"
                >
                  <MessageSquare className="w-3.5 h-3.5" /> Comment
                </button>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={onMerge}
                className="flex-1 flex items-center justify-center gap-1.5 py-2 bg-success/20 text-success border border-success/30 rounded-lg text-sm font-medium hover:bg-success/30 transition-colors"
              >
                <Merge className="w-4 h-4" /> Merge
              </button>
              <button
                onClick={onClosePR}
                className="px-4 py-2 bg-danger/20 text-danger border border-danger/30 rounded-lg text-sm hover:bg-danger/30 transition-colors"
              >
                Close
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
