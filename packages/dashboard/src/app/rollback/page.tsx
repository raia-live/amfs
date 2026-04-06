"use client";

import { useState, useEffect } from "react";
import { RotateCcw, Clock, Tag as TagIcon, AlertTriangle } from "lucide-react";
import { api, Tag } from "@/lib/api";

export default function RollbackPage() {
  const [mode, setMode] = useState<"timestamp" | "tag">("timestamp");
  const [timestamp, setTimestamp] = useState("");
  const [selectedTag, setSelectedTag] = useState("");
  const [tags, setTags] = useState<Tag[]>([]);
  const [result, setResult] = useState<{ entries_restored: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    api.listTags().then((d) => setTags(d.tags || [])).catch(() => {});
  }, []);

  async function handleRollback() {
    setError(null);
    setResult(null);
    try {
      let res;
      if (mode === "timestamp") {
        if (!timestamp) return;
        res = await api.rollback(timestamp);
      } else {
        if (!selectedTag) return;
        res = await api.rollbackToTag(selectedTag);
      }
      setResult(res);
      setConfirming(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Rollback failed");
      setConfirming(false);
    }
  }

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <RotateCcw className="w-5 h-5 text-warning" />
        <h1 className="text-xl font-semibold">Rollback</h1>
      </div>

      <div className="bg-warning/10 border border-warning/30 rounded-xl p-4 mb-6 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
        <div className="text-sm text-warning">
          Rollback restores agent memory to a previous point in time. Entries
          written after the target time will be superseded and the previous
          version will become live. This is recorded as a rollback event on the
          timeline.
        </div>
      </div>

      <div className="bg-void-dark border border-void-light rounded-xl p-6">
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setMode("timestamp")}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
              mode === "timestamp"
                ? "bg-warning/20 text-warning border border-warning/30"
                : "text-text-muted hover:text-text-primary hover:bg-void-light"
            }`}
          >
            <Clock className="w-4 h-4" />
            By Timestamp
          </button>
          <button
            onClick={() => setMode("tag")}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors ${
              mode === "tag"
                ? "bg-warning/20 text-warning border border-warning/30"
                : "text-text-muted hover:text-text-primary hover:bg-void-light"
            }`}
          >
            <TagIcon className="w-4 h-4" />
            By Tag
          </button>
        </div>

        {mode === "timestamp" ? (
          <div className="space-y-3">
            <label className="text-sm text-text-secondary block">
              Roll back to (ISO 8601)
            </label>
            <input
              type="datetime-local"
              value={timestamp}
              onChange={(e) => setTimestamp(e.target.value)}
              className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-warning/50"
            />
          </div>
        ) : (
          <div className="space-y-3">
            <label className="text-sm text-text-secondary block">
              Roll back to tag
            </label>
            {tags.length === 0 ? (
              <p className="text-sm text-text-muted">
                No tags available. Create a tag first.
              </p>
            ) : (
              <select
                value={selectedTag}
                onChange={(e) => setSelectedTag(e.target.value)}
                className="w-full bg-void-gray border border-void-light rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-warning/50"
              >
                <option value="">Select a tag…</option>
                {tags.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name} ({new Date(t.tagged_at).toLocaleDateString()})
                  </option>
                ))}
              </select>
            )}
          </div>
        )}

        <div className="mt-6">
          {!confirming ? (
            <button
              onClick={() => setConfirming(true)}
              disabled={mode === "timestamp" ? !timestamp : !selectedTag}
              className="w-full py-2.5 bg-warning/20 text-warning border border-warning/30 rounded-lg text-sm font-medium hover:bg-warning/30 disabled:opacity-40 transition-colors"
            >
              Prepare Rollback
            </button>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-danger text-center">
                Are you sure? This will restore memory to{" "}
                {mode === "timestamp" ? (
                  <span className="font-mono">{timestamp}</span>
                ) : (
                  <span className="font-medium">tag "{selectedTag}"</span>
                )}
                .
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => setConfirming(false)}
                  className="flex-1 py-2 bg-void-gray text-text-secondary border border-void-light rounded-lg text-sm hover:bg-void-light transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleRollback}
                  className="flex-1 py-2 bg-danger/20 text-danger border border-danger/30 rounded-lg text-sm font-medium hover:bg-danger/30 transition-colors"
                >
                  Confirm Rollback
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="mt-4 bg-success/10 border border-success/30 rounded-xl p-4 text-sm text-success">
          Rollback complete. {result.entries_restored} entries restored.
        </div>
      )}

      {error && (
        <div className="mt-4 bg-danger/10 border border-danger/30 rounded-xl p-4 text-sm text-danger">
          {error}
        </div>
      )}
    </div>
  );
}
