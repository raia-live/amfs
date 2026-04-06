"use client";

import { useState, useEffect } from "react";
import { GitBranch, Plus, Trash2, Merge, Eye, X } from "lucide-react";
import { api, Branch, DiffEntry } from "@/lib/api";

export default function BranchesPage() {
  const [branches, setBranches] = useState<Branch[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [diffBranch, setDiffBranch] = useState<string | null>(null);
  const [diff, setDiff] = useState<DiffEntry[]>([]);
  const [mergeTarget, setMergeTarget] = useState<string | null>(null);
  const [mergeStrategy, setMergeStrategy] = useState("fast_forward");

  async function fetchBranches() {
    setLoading(true);
    try {
      const data = await api.listBranches();
      setBranches(data.branches || []);
    } catch {
      setBranches([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchBranches();
  }, []);

  async function handleCreate(name: string, parent: string, desc: string) {
    await api.createBranch(name, parent, desc || undefined);
    setShowCreate(false);
    fetchBranches();
  }

  async function handleClose(name: string) {
    if (!confirm(`Close branch "${name}"? This cannot be undone.`)) return;
    await api.closeBranch(name);
    fetchBranches();
  }

  async function handleDiff(name: string) {
    setDiffBranch(name);
    try {
      const data = await api.diffBranch(name);
      setDiff(data.diff || []);
    } catch {
      setDiff([]);
    }
  }

  async function handleMerge() {
    if (!mergeTarget) return;
    try {
      await api.mergeBranch(mergeTarget, mergeStrategy);
      setMergeTarget(null);
      fetchBranches();
    } catch (e) {
      alert(e instanceof Error ? e.message : "Merge failed");
    }
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <GitBranch className="w-5 h-5 text-branch-blue" />
          <h1 className="text-xl font-semibold">Branches</h1>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 px-3 py-2 bg-branch-blue/20 text-branch-blue border border-branch-blue/30 rounded-lg text-sm hover:bg-branch-blue/30 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Branch
        </button>
      </div>

      {showCreate && (
        <CreateBranchModal
          onClose={() => setShowCreate(false)}
          onCreate={handleCreate}
        />
      )}

      {loading ? (
        <div className="text-center py-12 text-text-muted">Loading…</div>
      ) : branches.length === 0 ? (
        <div className="text-center py-12 text-text-muted">
          No branches yet. All activity is on <span className="text-sacred-gold">main</span>.
        </div>
      ) : (
        <div className="space-y-3">
          {branches.map((b) => (
            <div
              key={b.id || b.name}
              className="bg-void-dark border border-void-light rounded-xl p-4 flex items-start justify-between"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <GitBranch className="w-4 h-4 text-branch-blue" />
                  <span className="font-medium">{b.name}</span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      b.status === "open"
                        ? "bg-success/20 text-success"
                        : b.status === "merged"
                          ? "bg-branch-purple/20 text-branch-purple"
                          : "bg-danger/20 text-danger"
                    }`}
                  >
                    {b.status}
                  </span>
                </div>
                <p className="text-sm text-text-secondary">
                  from <span className="text-sacred-gold">{b.parent_branch}</span>
                  {b.description && <> — {b.description}</>}
                </p>
                <p className="text-xs text-text-muted mt-1">
                  Created {new Date(b.created_at).toLocaleDateString()} · {b.entry_count ?? 0} entries
                </p>
              </div>
              <div className="flex items-center gap-1.5 ml-4">
                <button
                  onClick={() => handleDiff(b.name)}
                  className="p-1.5 rounded-lg hover:bg-void-light text-text-muted hover:text-text-primary transition-colors"
                  title="View diff"
                >
                  <Eye className="w-4 h-4" />
                </button>
                {b.status === "open" && (
                  <>
                    <button
                      onClick={() => setMergeTarget(b.name)}
                      className="p-1.5 rounded-lg hover:bg-void-light text-text-muted hover:text-success transition-colors"
                      title="Merge"
                    >
                      <Merge className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleClose(b.name)}
                      className="p-1.5 rounded-lg hover:bg-void-light text-text-muted hover:text-danger transition-colors"
                      title="Close"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {diffBranch && (
        <DiffModal
          branch={diffBranch}
          diff={diff}
          onClose={() => setDiffBranch(null)}
        />
      )}

      {mergeTarget && (
        <MergeModal
          branch={mergeTarget}
          strategy={mergeStrategy}
          onStrategyChange={setMergeStrategy}
          onMerge={handleMerge}
          onClose={() => setMergeTarget(null)}
        />
      )}
    </div>
  );
}

function CreateBranchModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (name: string, parent: string, desc: string) => void;
}) {
  const [name, setName] = useState("");
  const [parent, setParent] = useState("main");
  const [desc, setDesc] = useState("");

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-void-dark border border-void-light rounded-xl p-6 w-full max-w-md">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Create Branch</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="space-y-3">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Branch name"
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-blue/50"
          />
          <input
            type="text"
            value={parent}
            onChange={(e) => setParent(e.target.value)}
            placeholder="Parent branch"
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-blue/50"
          />
          <textarea
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            placeholder="Description (optional)"
            rows={2}
            className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-branch-blue/50 resize-none"
          />
          <button
            onClick={() => onCreate(name, parent, desc)}
            disabled={!name}
            className="w-full py-2 bg-branch-blue/20 text-branch-blue border border-branch-blue/30 rounded-lg text-sm font-medium hover:bg-branch-blue/30 disabled:opacity-40 transition-colors"
          >
            Create Branch
          </button>
        </div>
      </div>
    </div>
  );
}

function DiffModal({
  branch,
  diff,
  onClose,
}: {
  branch: string;
  diff: DiffEntry[];
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-void-dark border border-void-light rounded-xl p-6 w-full max-w-2xl max-h-[80vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">
            Diff: <span className="text-branch-blue">{branch}</span> vs{" "}
            <span className="text-sacred-gold">main</span>
          </h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>
        {diff.length === 0 ? (
          <p className="text-text-muted text-sm">No differences found.</p>
        ) : (
          <div className="space-y-2">
            {diff.map((d, i) => (
              <div
                key={i}
                className="bg-void-gray rounded-lg p-3 border border-void-light"
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                      d.change_type === "added"
                        ? "bg-success/20 text-success"
                        : d.change_type === "modified"
                          ? "bg-warning/20 text-warning"
                          : "bg-danger/20 text-danger"
                    }`}
                  >
                    {d.change_type}
                  </span>
                  <span className="text-sm font-mono text-text-primary">
                    {d.entity_path}/{d.key}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div>
                    <span className="text-xs text-text-muted block mb-1">Branch</span>
                    <pre className="text-xs text-branch-blue bg-void-black/50 rounded p-1.5 max-h-20 overflow-auto">
                      {JSON.stringify(d.branch_value, null, 2) ?? "—"}
                    </pre>
                  </div>
                  <div>
                    <span className="text-xs text-text-muted block mb-1">Main</span>
                    <pre className="text-xs text-sacred-gold bg-void-black/50 rounded p-1.5 max-h-20 overflow-auto">
                      {JSON.stringify(d.main_value, null, 2) ?? "—"}
                    </pre>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MergeModal({
  branch,
  strategy,
  onStrategyChange,
  onMerge,
  onClose,
}: {
  branch: string;
  strategy: string;
  onStrategyChange: (s: string) => void;
  onMerge: () => void;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-void-dark border border-void-light rounded-xl p-6 w-full max-w-md">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">
            Merge <span className="text-branch-blue">{branch}</span> →{" "}
            <span className="text-sacred-gold">main</span>
          </h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="space-y-3">
          <div>
            <label className="text-sm text-text-secondary block mb-1">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => onStrategyChange(e.target.value)}
              className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-success/50"
            >
              <option value="fast_forward">Fast Forward</option>
              <option value="branch_wins">Branch Wins</option>
              <option value="main_wins">Main Wins</option>
            </select>
          </div>
          <button
            onClick={onMerge}
            className="w-full py-2 bg-success/20 text-success border border-success/30 rounded-lg text-sm font-medium hover:bg-success/30 transition-colors"
          >
            Merge Branch
          </button>
        </div>
      </div>
    </div>
  );
}
